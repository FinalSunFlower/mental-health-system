from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import DeclarativeBase, relationship
from datetime import datetime


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="counselor")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(20), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    gender = Column(String(10))
    age = Column(Integer)
    grade = Column(Integer)
    major = Column(String(100))
    college = Column(String(100))
    gpa = Column(Float)
    attendance_rate = Column(Float)
    phq9_scores = Column(JSON)
    gad7_scores = Column(JSON)
    pss10_scores = Column(JSON)
    cdrisc_scores = Column(JSON)
    mspss_scores = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    cuspnet_records = relationship("CuspNetRecord", back_populates="student", cascade="all, delete-orphan")


class CuspNetRecord(Base):
    __tablename__ = "cuspnet_records"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)

    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)

    global_a = Column(Float)
    global_b = Column(Float)
    global_c = Column(Float)
    resilience_reserve = Column(Float)
    critical_distance = Column(Float)
    tipping_point_warning = Column(Boolean, default=False)

    causal_adjacency = Column(JSON)
    centrality_ranking = Column(JSON)
    positive_feedback_loops = Column(JSON)
    attractor_states = Column(JSON)

    llm_primary = Column(JSON)
    llm_secondary = Column(JSON)
    llm_distortions = Column(JSON)
    llm_intervention = Column(JSON)
    llm_explanation = Column(JSON)

    model_version = Column(String(50), default="CuspNet-1.0")
    assessed_at = Column(DateTime, default=datetime.utcnow)

    student = relationship("Student", back_populates="cuspnet_records")
