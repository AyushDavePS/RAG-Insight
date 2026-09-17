import pytest

from rag_insight.generation import (
    ExperienceFact,
    employment_company_facts,
    experience_comparison_candidates,
    experience_comparison_facts,
    generate,
    grade,
    render_employment_company_answer,
    render_experience_comparison_answer,
    structured_problem_resolution_answer,
)
from rag_insight.models import Candidate, Chunk


class FakeLLM:
    def __init__(self, result):
        self.result = result

    def json(self, *args, **kwargs):
        return self.result


def test_unknown_citation_is_rejected():
    llm = FakeLLM({"claims": [{"text": "Claim", "source_ids": ["invented"]}]})
    context = [Candidate(Chunk("doc", "v1", "a.md", "Evidence", chunk_id="real"), 1)]
    with pytest.raises(ValueError, match="unknown source"):
        generate(llm, "Question", context)


def test_string_false_is_not_treated_as_true():
    context = [Candidate(Chunk("doc", "v1", "a.md", "Evidence", chunk_id="real"), 1)]
    with pytest.raises(ValueError, match="Malformed"):
        grade(FakeLLM({"sufficient": "false", "reason": "No evidence"}), "Question", context)


def test_grade_rejects_partial_document_wide_summary_before_calling_llm():
    context = [Candidate(Chunk("rohan", "v1", "rohan.pdf", "Rohan summary", chunk_id="rohan"), 1)]
    result = grade(FakeLLM({"sufficient": True, "reason": "Looks sufficient"}),
                   "List professional summaries from given documents", context, required_document_count=3)
    assert not result.sufficient
    assert "1 of 3" in result.reason


def test_grade_accepts_education_as_an_answer_to_what_named_person_studied():
    context = [
        Candidate(Chunk("aarav", "v1", "aarav.pdf", "Aarav Mehta\nBusiness Development Executive",
                        chunk_id="profile"), 1),
        Candidate(Chunk("aarav", "v1", "aarav.pdf", "EDUCATION\nBBA, Marketing — Gujarat University",
                        chunk_id="education", section_kind="education"), 0.9),
    ]
    result = grade(FakeLLM({"sufficient": False, "reason": "incorrect model verdict"}),
                   "What has Aarav studied?", context)
    assert result.sufficient
    assert "education section" in result.reason


def test_labeled_problem_and_resolution_answer_without_an_llm_grade_or_rewrite():
    class UnusedLLM:
        def json(self, *args, **kwargs):
            raise AssertionError("explicit labeled fields should not need an LLM")

    context = [Candidate(Chunk(
        "incident", "v1", "postmortem.pdf",
        "Incident\nRobots experienced route-planning resets.\nResolution\nFirmware 5.8.3 was deployed.\nCustomer impact\nLimited fleet impact.",
        chunk_id="incident",
    ), 1)]
    question = "Explain in short what the incident was about and how it was resolved"

    assessment = grade(UnusedLLM(), question, context)
    text, sources = generate(UnusedLLM(), question, context)

    assert assessment.sufficient
    assert text == ("Incident: Robots experienced route-planning resets. [1]\n"
                    "Resolution: Firmware 5.8.3 was deployed. [1]")
    assert [source.chunk_id for source in sources] == ["incident"]


def test_problem_resolution_route_does_not_bypass_a_requested_root_cause():
    context = [Candidate(Chunk(
        "incident", "v1", "postmortem.pdf",
        "Incident\nRobots experienced route-planning resets.\nResolution\nFirmware 5.8.3 was deployed.",
        chunk_id="incident",
    ), 1)]
    question = "What was the incident root cause and how was it resolved?"
    assert structured_problem_resolution_answer(question, context) is None


def test_document_wide_summary_generates_one_cited_claim_per_document():
    class PerDocumentLLM:
        def json(self, system, payload, **kwargs):
            evidence = payload["evidence"]
            return {"claims": [{"text": f"Summary for {evidence[0]['chunk_id']}",
                                "source_ids": [evidence[0]["chunk_id"]]}]}

    context = [
        Candidate(Chunk("isha", "v1", "isha.pdf", "Isha summary", chunk_id="isha"), 1),
        Candidate(Chunk("rohan", "v1", "rohan.pdf", "Rohan summary", chunk_id="rohan"), 0.9),
        Candidate(Chunk("aarav", "v1", "aarav.pdf", "Aarav summary", chunk_id="aarav"), 0.8),
    ]
    text, sources = generate(PerDocumentLLM(), "List professional summaries from given documents", context,
                             required_document_count=3)
    assert text == "Summary for isha [1]\n\nSummary for rohan [2]\n\nSummary for aarav [3]"
    assert [source.filename for source in sources] == ["isha.pdf", "rohan.pdf", "aarav.pdf"]


def test_role_differentiation_generates_grounded_contrasts_not_separate_summaries():
    class UnusedLLM:
        def json(self, *args, **kwargs):
            raise AssertionError("role evidence should be rendered deterministically")

    context = [
        Candidate(Chunk("sana", "v1", "sana.pdf", "Sana summary", chunk_id="sana", profile_role="aiml"), 1),
        Candidate(Chunk("arjun", "v1", "arjun.pdf", "Arjun summary", chunk_id="arjun",
                        profile_role="fullstack_aiml"), 0.9),
        Candidate(Chunk("vivek", "v1", "vivek.pdf", "Vivek summary", chunk_id="vivek", profile_role="aiml"), 0.8),
    ]
    text, sources = generate(UnusedLLM(), "Differentiate between Fullstack AIML and AIML candidates",
                             context, required_document_count=3)
    assert text.startswith("Documented role differences:")
    assert "Full Stack AI/ML" in text
    assert "AI/ML" in text
    assert [source.chunk_id for source in sources] == ["sana", "vivek", "arjun"]


def test_role_difference_recovers_from_an_unknown_model_citation_with_full_selected_scope():
    context = [
        Candidate(Chunk("sana", "v1", "sana.pdf", "Sana works on NLP.", chunk_id="sana"), 1),
        Candidate(Chunk("neha", "v1", "neha.pdf", "Neha builds React interfaces.", chunk_id="neha",
                        ), 0.9),
    ]
    llm = FakeLLM({"differences": [{
        "text": "AI/ML focuses on NLP while Frontend Development focuses on interfaces.",
        "source_ids": ["not-a-real-id"],
    }]})

    text, sources = generate(llm, "What is the difference between AIML and frontend candidates?", context,
                             required_document_count=2)

    assert "Grounded differences" in text
    assert "[1] [2]" in text
    assert [source.chunk_id for source in sources] == ["sana", "neha"]


def test_difference_generation_falls_back_to_document_groups_without_resume_metadata():
    class DifferenceLLM:
        def json(self, system, payload, **kwargs):
            assert set(payload["role_groups"]) == {"alpha.pdf", "beta.pdf"}
            return {"differences": [{
                "text": "alpha.pdf describes a caching policy, while beta.pdf describes a retention policy.",
                "source_ids": ["alpha", "beta"],
            }]}

    context = [
        Candidate(Chunk("alpha", "v1", "alpha.pdf", "Cache entries expire after one hour.", chunk_id="alpha"), 1),
        Candidate(Chunk("beta", "v1", "beta.pdf", "Records are retained for seven years.", chunk_id="beta"), 0.9),
    ]
    text, sources = generate(DifferenceLLM(), "What is the difference between these documents?", context,
                             required_document_count=2)
    assert text.startswith("Grounded differences:\n- alpha.pdf describes a caching policy")
    assert [source.chunk_id for source in sources] == ["alpha", "beta"]


def test_comparative_generation_adds_coverage_citations_when_model_cites_only_winner():
    class ComparativeLLM:
        def json(self, system, payload, **kwargs):
            assert {item["document_id"] for item in payload["candidates"]} == {"isha", "rohan", "aarav"}
            return {
                "selected_document_id": "isha",
                "criteria": ["relevant experience", "documented outcomes"],
                "rationale": "Isha has the broadest documented business-development experience.",
                "supporting_source_ids": ["isha"],
            }

    context = [
        Candidate(Chunk("isha", "v1", "isha.pdf", "Isha has four years.", chunk_id="isha"), 1),
        Candidate(Chunk("rohan", "v1", "rohan.pdf", "Rohan has 1.5 years.", chunk_id="rohan"), 0.9),
        Candidate(Chunk("aarav", "v1", "aarav.pdf", "Aarav has 2.8 years.", chunk_id="aarav"), 0.8),
    ]
    text, sources = generate(ComparativeLLM(), "Which candidate profile is best?", context,
                             required_document_count=3, comparative=True)
    assert text.startswith("Recommended candidate: Isha has four years.")
    assert "evidence-based inference" in text
    assert text.endswith("[1] [2] [3]")
    assert [source.chunk_id for source in sources] == ["isha", "rohan", "aarav"]


def test_comparative_generation_rejects_unknown_selected_document():
    llm = FakeLLM({
        "selected_document_id": "invented",
        "criteria": ["experience"],
        "rationale": "Unsupported selection.",
        "supporting_source_ids": ["isha"],
    })
    context = [
        Candidate(Chunk("isha", "v1", "isha.pdf", "Isha evidence", chunk_id="isha"), 1),
        Candidate(Chunk("rohan", "v1", "rohan.pdf", "Rohan evidence", chunk_id="rohan"), 0.9),
    ]
    with pytest.raises(ValueError, match="unknown document"):
        generate(llm, "Which candidate is best?", context, required_document_count=2, comparative=True)


def test_highest_experience_comparison_is_deterministic_and_cites_every_candidate():
    class UnusedLLM:
        def json(self, *args, **kwargs):
            raise AssertionError("numeric experience comparison should not call the model")

    context = [
        Candidate(Chunk("aarav", "v1", "aarav.pdf", "Aarav Mehta\n2.8 years of experience",
                        chunk_id="aarav"), 1),
        Candidate(Chunk("isha", "v1", "isha.pdf", "Isha Shah\n4 years of experience",
                        chunk_id="isha"), 0.9),
        Candidate(Chunk("karan", "v1", "karan.pdf", "Karan Singh\n6 years of experience",
                        chunk_id="karan"), 0.8),
    ]
    text, sources = generate(UnusedLLM(), "candidate with highest experince", context,
                             required_document_count=3, comparative=True)
    assert text == "Karan Singh has the highest stated experience: 6 years. [1] [2] [3]"
    assert [source.chunk_id for source in sources] == ["karan", "isha", "aarav"]


def test_experience_extrema_renders_highest_and_lowest_from_complete_facts():
    facts = [
        ExperienceFact("aarav", "Aarav Mehta", 2.8, [Chunk("aarav", "v1", "aarav.pdf", "2.8 years", chunk_id="aarav")], False),
        ExperienceFact("isha", "Isha Shah", 4.0, [Chunk("isha", "v1", "isha.pdf", "4 years", chunk_id="isha")], False),
        ExperienceFact("rohan", "Rohan Patel", 1.5, [Chunk("rohan", "v1", "rohan.pdf", "1.5 years", chunk_id="rohan")], False),
    ]
    text, sources = render_experience_comparison_answer("BDEs with highest and lowest experience", facts)
    assert "Isha Shah has the highest stated experience: 4 years." in text
    assert "Rohan Patel has the lowest stated experience: 1.5 years." in text
    assert [source.chunk_id for source in sources] == ["isha", "aarav", "rohan"]


def test_experience_map_reduce_keeps_one_numeric_fact_per_document():
    candidates = [
        Candidate(Chunk("aarav", "v1", "aarav.pdf", "Aarav\n2.8 years of experience", chunk_id="aarav"), 1),
        Candidate(Chunk("isha", "v1", "isha.pdf", "Isha\n4 years of experience", chunk_id="isha"), 1),
        Candidate(Chunk("neha", "v1", "neha.pdf", "Neha\n3.5 years of experience", chunk_id="neha"), 1),
        Candidate(Chunk("sana", "v1", "sana.pdf", "Sana\n3 years of experience", chunk_id="sana"), 1),
        Candidate(Chunk("rohan", "v1", "rohan.pdf", "Rohan\n1.5 years of experience", chunk_id="rohan"), 1),
    ]
    facts = experience_comparison_candidates("Who has the highest years of experience?", candidates)
    assert [fact.chunk.chunk_id for fact in facts] == ["aarav", "isha", "neha", "rohan", "sana"]
    assert [fact.score for fact in facts] == [2.8, 4.0, 3.5, 1.5, 3.0]


def test_experience_map_reduce_falls_back_to_non_overlapping_dated_roles_and_composes_count():
    maya_profile = Candidate(Chunk("maya", "v1", "maya.pdf", "Maya Joshi\nFull Stack Developer — AIML",
                                   chunk_id="maya-profile"), 1)
    maya_work = Candidate(Chunk("maya", "v1", "maya.pdf",
                                "Engineer — BuildLab | Noida | Jan 2021 - Jun 2025",
                                chunk_id="maya-work"), 1)
    sana = Candidate(Chunk("sana", "v1", "sana.pdf", "Sana Khan\nAI/ML Engineer\n"
                           "3 years of experience", chunk_id="sana-profile"), 1)
    question = "How many Fullstack AIML candidates are there and who has the highest experience?"
    facts = experience_comparison_facts(question, [maya_work, sana, maya_profile])
    assert [(fact.label, fact.years, fact.derived_from_dates) for fact in facts] == [
        ("Maya Joshi", 4.5, True), ("Sana Khan", 3.0, False),
    ]
    text, sources = render_experience_comparison_answer(question, facts)
    assert text.startswith("There are 2 candidates in scope. Maya Joshi has the highest documented employment duration")
    assert [source.chunk_id for source in sources] == ["maya-profile", "maya-work", "sana-profile"]


def test_employment_map_reduce_counts_companies_per_document_and_ignores_education():
    candidates = [
        Candidate(Chunk("aarav", "v1", "aarav.pdf", "Aarav Mehta\nBusiness Development Executive",
                        chunk_id="aarav-profile"), 1),
        Candidate(Chunk("aarav", "v1", "aarav.pdf",
                        "BDE — CloudNexa Systems | Ahmedabad | 2023\n"
                        "Associate — LeadOrbit | Ahmedabad | 2021\n"
                        "BBA — Gujarat University, 2020", chunk_id="aarav-work"), 1),
        Candidate(Chunk("isha", "v1", "isha.pdf", "Isha Shah\nBusiness Development Executive",
                        chunk_id="isha-profile"), 1),
        Candidate(Chunk("isha", "v1", "isha.pdf", "BDE — VertexWave Technologies | Mumbai | 2022",
                        chunk_id="isha-work"), 1),
    ]
    facts = employment_company_facts("Candidates who have worked in more than 1 company", candidates)

    assert [(fact.label, fact.companies) for fact in facts] == [
        ("Aarav Mehta", ["CloudNexa Systems", "LeadOrbit"]),
        ("Isha Shah", ["VertexWave Technologies"]),
    ]
    text, sources = render_employment_company_answer(facts, 1)
    assert text == "Aarav Mehta: CloudNexa Systems, LeadOrbit (2 companies) [1] [2]"
    assert [source.chunk_id for source in sources] == ["aarav-profile", "aarav-work"]


def test_table_question_transcribes_extracted_rows_without_an_llm_rewrite():
    context = [Candidate(Chunk("doc", "v1", "table.pdf", "ID Name Role\n001 Jane Developer\n002 Sam Designer\nText after table", chunk_id="table"), 1)]
    text, sources = generate(FakeLLM({}), "What values are inside the table?", context)
    assert text == "ID Name Role\n001 Jane Developer\n002 Sam Designer [1]"
    assert [source.chunk_id for source in sources] == ["table"]


def test_version_year_question_transcribes_all_retrieved_pairs():
    first = Candidate(Chunk("doc", "v1", "history.pdf", "PDF 1.0\n1993\nPDF 1.7\n2006", chunk_id="first"), 1)
    second = Candidate(Chunk("doc", "v1", "history.pdf", "PDF 1.7\n2006\nPDF 2.0\n2017", chunk_id="second"), 0.9)
    text, sources = generate(FakeLLM({}), "List all PDF versions and years", [first, second])
    assert text == "PDF 1.0 1993\nPDF 1.7 2006\nPDF 2.0 2017 [1] [2]"
    assert [source.chunk_id for source in sources] == ["first", "second"]


def test_regional_distribution_transcribes_every_vertical_table_row():
    text = "Region\nUsers\nShare (%)\nAvg. Pages/Visit\nTop Browser\nNorth America\n18,400\n29.5\n3.2\nChrome\nEurope\n16,200\n26.0\n3.5\nChrome\nTable 6.2"
    context = [Candidate(Chunk("doc", "v1", "regions.pdf", text, chunk_id="regions"), 1)]
    answer, sources = generate(FakeLLM({}), "List all regional distribution data", context)
    assert answer == "Region | Users | Share (%) | Avg. Pages/Visit | Top Browser\nNorth America | 18,400 | 29.5 | 3.2 | Chrome\nEurope | 16,200 | 26.0 | 3.5 | Chrome [1]"
    assert [source.chunk_id for source in sources] == ["regions"]
