import pytest

from rag_insight.models import Candidate, Chunk
from rag_insight.retrieval import (
    bm25_search,
    dense_search,
    filter_rows_by_profile,
    fuse,
    is_comparative_query,
    is_structured_aggregation_request,
    requested_profile_roles,
    requests_role_differentiation,
    requires_document_coverage,
    select_context,
)


def chunk(key, text):
    return Chunk("doc", "v1", "test.md", text, chunk_id=key)


def test_rrf_merges_duplicate_ids_and_ignores_score_scales():
    a, b = chunk("a", "Redis"), chunk("b", "PostgreSQL")
    result = fuse([[Candidate(a, 0.2), Candidate(b, 0.1)], [Candidate(b, 9000)]])
    assert len(result) == 2
    assert result[0].chunk.chunk_id == "b"


def test_bm25_finds_identifier_and_excludes_zero_matches():
    rows = [(chunk("a", "Send X-API-Key header"), []), (chunk("b", "Redis caching"), [])]
    assert [c.chunk.chunk_id for c in bm25_search("X-API-Key", rows, 5)] == ["a"]


def test_dense_search_rejects_vector_dimension_mismatch():
    rows = [(chunk("a", "Redis"), [1.0, 0.0])]
    with pytest.raises(ValueError, match="dimensions do not match"):
        dense_search([1.0, 0.0, 0.0], rows, 5)


def test_context_prefers_a_specific_multi_term_match():
    table = chunk("table", "Tables and Data Display\n001 Jane Smith")
    generic = chunk("generic", "This document includes tables, images, and hyperlinks")
    settings = type("Settings", (), {"mmr": False, "context_k": 5, "context_tokens": 100,
                                     "mmr_lambda": 0.7})()
    tokenizer = type("Tokenizer", (), {"encode": lambda _, text, add_special_tokens=False: text.split()})()
    selected = select_context([Candidate(table, 1), Candidate(generic, 0.9)],
                              [(table, [1]), (generic, [1])], settings, tokenizer,
                              "what values are present inside the tables and displays?")
    assert [candidate.chunk.chunk_id for candidate in selected] == ["table"]


def test_context_does_not_turn_paris_into_an_unrelated_lexical_match():
    paris = chunk("paris", "Paris is the capital and largest city of France.")
    luang_prabang = chunk("laos", "Located at the confluence of two rivers, Luang Prabang is in Laos.")
    settings = type("Settings", (), {"mmr": False, "context_k": 5, "context_tokens": 100,
                                     "mmr_lambda": 0.7})()
    tokenizer = type("Tokenizer", (), {"encode": lambda _, text, add_special_tokens=False: text.split()})()
    selected = select_context([Candidate(paris, 1), Candidate(luang_prabang, 0.9)],
                              [(paris, [1]), (luang_prabang, [1])], settings, tokenizer,
                              "Where is Paris located?")
    assert [candidate.chunk.chunk_id for candidate in selected] == ["paris", "laos"]


def test_candidate_summary_context_covers_distinct_documents():
    isha = Chunk("isha", "v1", "isha.pdf", "Isha Shah is a client-facing BDE.", chunk_id="isha")
    isha_experience = Chunk("isha", "v1", "isha.pdf", "Isha manages partnerships.", chunk_id="isha-experience")
    aarav = Chunk("aarav", "v1", "aarav.pdf", "Aarav Mehta has BDE experience.", chunk_id="aarav")
    rohan = Chunk("rohan", "v1", "rohan.pdf", "Rohan Patel sells industrial products.", chunk_id="rohan")
    settings = type("Settings", (), {"mmr": False, "context_k": 3, "context_tokens": 100,
                                     "mmr_lambda": 0.7})()
    tokenizer = type("Tokenizer", (), {"encode": lambda _, text, add_special_tokens=False: text.split()})()
    candidates = [Candidate(isha, 1), Candidate(isha_experience, 0.9), Candidate(rohan, 0.8), Candidate(aarav, 0.7)]
    selected = select_context(candidates, [(candidate.chunk, [1]) for candidate in candidates], settings, tokenizer,
                              "summary of BDE candidates")
    assert [candidate.chunk.document_id for candidate in selected] == ["isha", "rohan", "aarav"]


def test_document_summary_context_covers_distinct_documents_without_candidate_wording():
    rohan = Chunk("rohan", "v1", "rohan.pdf", "Rohan professional summary.", chunk_id="rohan")
    rohan_details = Chunk("rohan", "v1", "rohan.pdf", "Rohan experience details.", chunk_id="rohan-details")
    isha = Chunk("isha", "v1", "isha.pdf", "Isha professional summary.", chunk_id="isha")
    aarav = Chunk("aarav", "v1", "aarav.pdf", "Aarav professional summary.", chunk_id="aarav")
    settings = type("Settings", (), {"mmr": False, "context_k": 3, "context_tokens": 100,
                                     "mmr_lambda": 0.7})()
    tokenizer = type("Tokenizer", (), {"encode": lambda _, text, add_special_tokens=False: text.split()})()
    candidates = [Candidate(rohan, 1), Candidate(rohan_details, 0.9), Candidate(isha, 0.8), Candidate(aarav, 0.7)]
    selected = select_context(candidates, [(candidate.chunk, [1]) for candidate in candidates], settings, tokenizer,
                              "List professional summaries from given documents")
    assert [candidate.chunk.document_id for candidate in selected] == ["rohan", "isha", "aarav"]


def test_bde_scope_filter_excludes_frontend_profile_before_ranking():
    isha = Chunk("isha", "v1", "isha.pdf", "Business Development Executive", chunk_id="isha",
                 profile_role="business_development")
    rohan = Chunk("rohan", "v1", "rohan.pdf", "Business Development Executive", chunk_id="rohan",
                  profile_role="business_development")
    neha = Chunk("neha", "v1", "neha.pdf", "Frontend Developer", chunk_id="neha",
                 profile_role="frontend_development")
    rows, roles = filter_rows_by_profile([(isha, [1]), (rohan, [1]), (neha, [1])],
                                         "List summaries of only the BDE candidates")
    assert roles == {"business_development"}
    assert [chunk.filename for chunk, _ in rows] == ["isha.pdf", "rohan.pdf"]


def test_profile_scope_matches_a_stored_generic_title():
    analyst = Chunk("analyst", "v1", "analyst.pdf", "Data Analyst", chunk_id="analyst", profile_role="data_analyst")
    developer = Chunk("developer", "v1", "developer.pdf", "Backend Developer", chunk_id="developer",
                      profile_role="backend_developer")
    rows, roles = filter_rows_by_profile([(analyst, [1]), (developer, [1])], "Summarize only data analyst candidates")
    assert roles == {"data_analyst"}
    assert [chunk.filename for chunk, _ in rows] == ["analyst.pdf"]


def test_aiml_role_scope_includes_fullstack_aiml_and_aiml_but_excludes_other_roles():
    maya = Chunk("maya", "v1", "maya.pdf", "Full Stack Developer — AI/ML", chunk_id="maya",
                 profile_role="fullstack_aiml")
    daniel = Chunk("daniel", "v1", "daniel.pdf", "Fullstack AI/ML Engineer", chunk_id="daniel",
                   profile_role="fullstack_aiml")
    sana = Chunk("sana", "v1", "sana.pdf", "AI/ML Engineer", chunk_id="sana", profile_role="aiml")
    bde = Chunk("rohan", "v1", "rohan.pdf", "Business Development Executive", chunk_id="rohan",
                profile_role="business_development")
    rows, roles = filter_rows_by_profile(
        [(maya, [1]), (daniel, [1]), (sana, [1]), (bde, [1])],
        "List out the Fullstack AIML and just AIML candidates.",
    )
    assert roles == {"fullstack_aiml", "aiml"}
    assert [chunk.filename for chunk, _ in rows] == ["maya.pdf", "daniel.pdf", "sana.pdf"]


def test_fullstack_aiml_vs_frontend_does_not_accidentally_include_standalone_aiml():
    maya = Chunk("maya", "v1", "maya.pdf", "Full Stack Developer — AI/ML", chunk_id="maya",
                 profile_role="fullstack_aiml")
    neha = Chunk("neha", "v1", "neha.pdf", "Frontend Developer", chunk_id="neha",
                 profile_role="frontend_development")
    karan = Chunk("karan", "v1", "karan.pdf", "Frontend Developer", chunk_id="karan",
                  profile_role="frontend_development")
    sana = Chunk("sana", "v1", "sana.pdf", "AI/ML Engineer", chunk_id="sana", profile_role="aiml")
    rows, roles = filter_rows_by_profile(
        [(maya, [1]), (neha, [1]), (karan, [1]), (sana, [1])],
        "List out the difference between the Fullstack AIML and the frontend candidate.",
    )
    assert roles == {"fullstack_aiml", "frontend_development"}
    assert [chunk.filename for chunk, _ in rows] == ["maya.pdf", "neha.pdf", "karan.pdf"]


def test_plural_bdes_is_a_hard_business_development_role_scope():
    query = "BDEs with highest and lowest experience"
    assert requested_profile_roles(query) == {"business_development"}
    assert requires_document_coverage(query)


def test_summary_request_prefers_summary_section_over_higher_ranked_experience():
    experience = Chunk("rohan", "v1", "rohan.pdf", "PROFESSIONAL EXPERIENCE", chunk_id="experience",
                       profile_role="business_development", section_kind="professional_experience")
    summary = Chunk("rohan", "v1", "rohan.pdf", "PROFESSIONAL SUMMARY", chunk_id="summary",
                    profile_role="business_development", section_kind="professional_summary")
    settings = type("Settings", (), {"mmr": False, "context_k": 1, "context_tokens": 100,
                                     "mmr_lambda": 0.7})()
    tokenizer = type("Tokenizer", (), {"encode": lambda _, text, add_special_tokens=False: text.split()})()
    selected = select_context([Candidate(experience, 1), Candidate(summary, 0.9)],
                              [(experience, [1]), (summary, [1])], settings, tokenizer,
                              "List the professional summary of the BDE candidate")
    assert [candidate.chunk.chunk_id for candidate in selected] == ["summary"]


def test_role_difference_selects_summary_and_skills_evidence_for_each_role():
    maya_summary = Chunk("maya", "v1", "maya.pdf", "Maya\nFull Stack AI/ML Engineer\nPROFESSIONAL SUMMARY",
                         chunk_id="maya-summary", profile_role="fullstack_aiml", section_kind="professional_summary")
    maya_work = Chunk("maya", "v1", "maya.pdf", "PROFESSIONAL EXPERIENCE\nReact and FastAPI",
                      chunk_id="maya-work", profile_role="fullstack_aiml", section_kind="professional_experience")
    maya_skills = Chunk("maya", "v1", "maya.pdf", "CORE SKILLS\nReact FastAPI PyTorch",
                        chunk_id="maya-skills", profile_role="fullstack_aiml", section_kind="skills")
    sana_summary = Chunk("sana", "v1", "sana.pdf", "Sana\nAI/ML Engineer\nPROFESSIONAL SUMMARY",
                         chunk_id="sana-summary", profile_role="aiml", section_kind="professional_summary")
    sana_work = Chunk("sana", "v1", "sana.pdf", "PROFESSIONAL EXPERIENCE\nOCR pipeline",
                      chunk_id="sana-work", profile_role="aiml", section_kind="professional_experience")
    sana_skills = Chunk("sana", "v1", "sana.pdf", "CORE SKILLS\nPyTorch OCR Transformers",
                        chunk_id="sana-skills", profile_role="aiml", section_kind="skills")
    settings = type("Settings", (), {"mmr": False, "context_k": 4, "context_tokens": 100,
                                     "mmr_lambda": 0.7})()
    tokenizer = type("Tokenizer", (), {"encode": lambda _, text, add_special_tokens=False: text.split()})()
    query = "List the difference between the Fullstack AIML candidate and AIML candidate"
    assert requests_role_differentiation(query)
    selected = select_context(
        [Candidate(maya_work, 1), Candidate(sana_work, 0.9), Candidate(maya_summary, 0.8),
         Candidate(sana_summary, 0.7), Candidate(maya_skills, 0.6), Candidate(sana_skills, 0.5)],
        [(maya_work, [1]), (sana_work, [1]), (maya_summary, [1]), (sana_summary, [1]),
         (maya_skills, [1]), (sana_skills, [1])],
        settings, tokenizer, query,
    )
    assert [candidate.chunk.chunk_id for candidate in selected] == [
        "maya-summary", "sana-summary", "maya-skills", "sana-skills",
    ]


def test_context_boosts_generic_education_section_in_explicitly_named_document():
    aarav_profile = Chunk("aarav", "v1", "aarav.pdf", "Aarav Mehta\nCORE SKILLS\nQualification basics",
                          chunk_id="profile", section_kind="skills")
    aarav_education = Chunk("aarav", "v1", "aarav.pdf", "EDUCATION\nBBA, Marketing — Gujarat University",
                            chunk_id="education", section_kind="education")
    rohan_education = Chunk("rohan", "v1", "rohan.pdf", "EDUCATION\nB.E. Mechanical Engineering",
                            chunk_id="rohan-education", section_kind="education")
    settings = type("Settings", (), {"mmr": False, "context_k": 1, "context_tokens": 100,
                                     "mmr_lambda": 0.7})()
    tokenizer = type("Tokenizer", (), {"encode": lambda _, text, add_special_tokens=False: text.split()})()
    selected = select_context(
        [Candidate(aarav_profile, 1), Candidate(rohan_education, 0.9), Candidate(aarav_education, 0.8)],
        [(aarav_profile, [1]), (rohan_education, [1]), (aarav_education, [1])], settings, tokenizer,
        "What has Aarav studied?",
    )
    assert [candidate.chunk.chunk_id for candidate in selected] == ["education"]


def test_collection_wide_company_list_requires_document_coverage_and_experience_chunks():
    isha = Chunk("isha", "v1", "isha.pdf", "PROFESSIONAL EXPERIENCE\nVertexWave Technologies",
                 chunk_id="isha-work", section_kind="professional_experience")
    rohan = Chunk("rohan", "v1", "rohan.pdf", "PROFESSIONAL EXPERIENCE\nAeroFab Components",
                  chunk_id="rohan-work", section_kind="professional_experience")
    aarav = Chunk("aarav", "v1", "aarav.pdf", "WORK EXPERIENCE\nCloudNexa Systems",
                  chunk_id="aarav-work", section_kind="professional_experience")
    settings = type("Settings", (), {"mmr": False, "context_k": 3, "context_tokens": 100,
                                     "mmr_lambda": 0.7})()
    tokenizer = type("Tokenizer", (), {"encode": lambda _, text, add_special_tokens=False: text.split()})()
    query = "List the companies where the BDE candidates worked"
    assert requires_document_coverage(query)
    selected = select_context([Candidate(isha, 1), Candidate(rohan, 0.9), Candidate(aarav, 0.8)],
                              [(isha, [1]), (rohan, [1]), (aarav, [1])], settings, tokenizer, query)
    assert [candidate.chunk.document_id for candidate in selected] == ["isha", "rohan", "aarav"]


def test_highest_experience_candidate_is_a_collection_wide_comparison():
    query = "Give me the candidate with highest experince"
    assert requires_document_coverage(query)
    assert is_comparative_query(query)


def test_semantic_intent_can_recognize_an_indirect_comparison_phrase():
    intent = {"collection_wide": True, "comparative": True}
    query = "Who has spent the longest time in the field?"
    assert requires_document_coverage(query, intent)
    assert is_comparative_query(query, intent)


def test_difference_question_requires_document_coverage_without_agent_classification():
    assert requires_document_coverage("What is the difference between frontend and AIML candidates?")


def test_aggregation_intent_broadens_coverage_without_a_keyword_heuristic_match():
    query = "Which applicants satisfy the tenure threshold?"
    assert not requires_document_coverage(query)
    assert requires_document_coverage(query, {"aggregation": True})
    assert is_structured_aggregation_request("How many candidate profiles have more than one employer?")


def test_aggregate_context_keeps_same_page_continuation():
    first = Chunk("doc", "v1", "versions.pdf", "PDF version and year", page=11, chunk_id="first")
    continuation = Chunk("doc", "v1", "versions.pdf", "PDF 2.0 2017", page=11, chunk_id="continuation")
    other = Chunk("doc", "v1", "versions.pdf", "Unrelated page", page=12, chunk_id="other")
    settings = type("Settings", (), {"mmr": False, "context_k": 2, "context_tokens": 100,
                                     "mmr_lambda": 0.7})()
    tokenizer = type("Tokenizer", (), {"encode": lambda _, text, add_special_tokens=False: text.split()})()
    selected = select_context([Candidate(first, 1), Candidate(other, 0.9), Candidate(continuation, 0.1)],
                              [(first, [1]), (continuation, [1]), (other, [1])], settings, tokenizer,
                              "list all PDF versions and years")
    assert [candidate.chunk.chunk_id for candidate in selected] == ["first", "continuation"]
