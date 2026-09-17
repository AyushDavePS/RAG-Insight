from rag_insight.models import Chunk
from rag_insight.registry import build_document_registry


def test_registry_derives_resume_role_sections_and_skills_from_document_chunks():
    chunks = [
        Chunk("sana", "v1", "sana.pdf", "Sana Khan\nAI/ML Engineer\nPROFESSIONAL SUMMARY\nBuilds NLP systems.",
              chunk_id="summary"),
        Chunk("sana", "v1", "sana.pdf", "CORE SKILLS\nPython | PyTorch | RAG\nPROFESSIONAL EXPERIENCE",
              chunk_id="skills"),
    ]
    [record] = build_document_registry(chunks)
    assert record.document_type == "resume_profile"
    assert record.profile_role == "aiml"
    assert record.summary_chunk_id == "summary"
    assert record.skills == ("Python", "PyTorch", "RAG")
    assert set(record.sections) == {"professional_summary", "professional_experience", "skills"}


def test_registry_classifies_generic_job_description_without_profile_metadata():
    chunks = [
        Chunk("job", "v1", "job.md", "Senior Developer\nRESPONSIBILITIES\nBuild services.\nREQUIREMENTS\nPython.",
              chunk_id="job"),
    ]
    [record] = build_document_registry(chunks)
    assert record.document_type == "job_description"
    assert record.profile_role == ""
    assert record.target_role == "senior_developer"
    assert set(record.sections) == {"requirements", "responsibilities"}
