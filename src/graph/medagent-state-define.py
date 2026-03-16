"""
ClinicalState — central TypedDict flowing through the LangGraph clinical workflow.

Every agent reads from and writes to fields in this state. The graph passes
the state object through nodes sequentially; each node returns a partial dict
that gets merged back.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class VitalSigns(TypedDict, total=False):
    heart_rate: int
    blood_pressure: str  # e.g. "120/80"
    temperature: float  # Celsius
    respiratory_rate: int
    spo2: int  # percentage


class DiagnosisEntry(TypedDict):
    diagnosis: str
    probability: float  # 0.0 – 1.0
    reasoning: str
    supporting_evidence: list[str]


class DrugInteraction(TypedDict):
    drug_pair: list[str]
    severity: str  # "major" | "moderate" | "minor"
    description: str


class LabInterpretation(TypedDict):
    test_name: str
    value: float
    unit: str
    flag: str  # "normal" | "high" | "low" | "critical"
    clinical_note: str


class LiteratureResult(TypedDict):
    title: str
    snippet: str
    relevance_score: float
    source: str


class ClinicalState(TypedDict, total=False):
    """Full state schema for the MedAgent clinical workflow graph."""

    # ── Input ──────────────────────────────────────────────────────────
    patient_input: str  # Raw clinical vignette text

    # ── Triage outputs ─────────────────────────────────────────────────
    esi_level: int  # Emergency Severity Index 1-5
    triage_reasoning: str
    chief_complaint: str
    vitals: VitalSigns
    medications: list[str]
    lab_results: list[dict]  # Raw lab name/value/unit dicts

    # ── Literature outputs ─────────────────────────────────────────────
    literature_results: list[LiteratureResult]
    literature_summary: str

    # ── Reasoning outputs ──────────────────────────────────────────────
    differential_diagnosis: list[DiagnosisEntry]
    recommended_workup: list[str]
    drug_interactions: list[DrugInteraction]
    lab_interpretations: list[LabInterpretation]

    # ── Report outputs ─────────────────────────────────────────────────
    soap_note: str

    # ── Meta / control ─────────────────────────────────────────────────
    messages: Annotated[list, add_messages]
    human_approved: bool
    error: str | None
