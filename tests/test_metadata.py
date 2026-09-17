from rag_insight.metadata import enrich_chunks
from rag_insight.models import Chunk


def test_resume_metadata_is_shared_across_chunks_and_preserves_section_type():
    first = Chunk("isha", "v1", "isha.pdf", "Isha Shah\nBusiness Development Executive\nPROFESSIONAL SUMMARY",
                  chunk_id="summary")
    second = Chunk("isha", "v1", "isha.pdf", "PROFESSIONAL EXPERIENCE\nVertexWave Technologies", chunk_id="experience")
    enriched = enrich_chunks([first, second])
    assert [chunk.profile_role for chunk in enriched] == ["business_development", "business_development"]
    assert [chunk.section_kind for chunk in enriched] == ["professional_summary", "professional_experience"]


def test_document_role_survives_unordered_stored_chunks_before_the_title_chunk():
    detail = Chunk("aarav", "v1", "aarav.pdf", "EDUCATION\nBBA\nMarketing\nUniversity\n2023\n"
                   "ADDITIONAL INFORMATION\nEnglish\nHindi\nAvailable", chunk_id="detail")
    profile = Chunk("aarav", "v1", "aarav.pdf", "Aarav Mehta\nBusiness Development Executive\n"
                    "PROFESSIONAL SUMMARY", chunk_id="profile")
    enriched = enrich_chunks([detail, profile])
    assert [chunk.profile_role for chunk in enriched] == ["business_development", "business_development"]


def test_unknown_resume_titles_keep_a_conservative_normalized_role_label():
    chunk = Chunk("analyst", "v1", "analyst.pdf", "Maya Rao\nData Analyst\nPROFESSIONAL SUMMARY", chunk_id="analyst")
    assert enrich_chunks([chunk])[0].profile_role == "data_analyst"


def test_compound_fullstack_aiml_and_aiml_titles_have_distinct_normalized_roles():
    fullstack = Chunk("maya", "v1", "maya.pdf", "Maya Joshi\nFull Stack Developer — AI/ML",
                      chunk_id="fullstack")
    aiml = Chunk("sana", "v1", "sana.pdf", "Sana Khan\nAI/ML Engineer", chunk_id="aiml")
    enriched = enrich_chunks([fullstack, aiml])
    assert [chunk.profile_role for chunk in enriched] == ["fullstack_aiml", "aiml"]


def test_generic_section_metadata_recognizes_resume_and_job_description_headings():
    education = Chunk("resume", "v1", "resume.pdf", "EDUCATION\nBBA, Marketing", chunk_id="education")
    requirements = Chunk("job", "v1", "job.pdf", "REQUIREMENTS\n3 years of sales experience", chunk_id="requirements")
    employment = Chunk("work", "v1", "work.pdf", "EMPLOYMENT HISTORY\nExample Corp", chunk_id="employment")
    skills_prose = Chunk("notes", "v1", "notes.txt", "Qualification happens after the call.", chunk_id="prose")
    enriched = enrich_chunks([education, requirements, employment, skills_prose])
    assert [chunk.section_kind for chunk in enriched] == ["education", "requirements", "professional_experience", ""]
