from sqlalchemy import JSON, Column, Float, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Experiment(Base):
    """A PTQ experiment with results."""

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    group_name = Column(String)
    model_name = Column(String, nullable=False)
    task = Column(String, nullable=False)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    ptq_method = Column(String, nullable=False)
    activation_scale_rule = Column(String, nullable=False)
    weight_scale_rule = Column(String, nullable=False)
    smoothquant_alpha = Column(Float, nullable=True)
    results = Column(JSON, nullable=False)
