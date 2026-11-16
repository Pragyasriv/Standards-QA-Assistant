"""Configurable names for supported regulations, unsupported instruments, and overlap topics.

Aliases are matched as whole phrases. Overlap topics such as lead and CE marking
do not select RoHS or REACH by themselves.
"""

from pydantic import BaseModel, ConfigDict, Field


class NamedAliases(BaseModel):
    """One instrument and the phrases that name it."""

    model_config = ConfigDict(extra="forbid")

    name: str
    aliases: list[str] = Field(min_length=1)


class AmbiguityTopic(BaseModel):
    """A phrase that can belong to more than one instrument."""

    model_config = ConfigDict(extra="forbid")

    topic: str
    phrases: list[str] = Field(min_length=1)
    clarification: str


class GuardrailCatalog(BaseModel):
    """Phrase lists used by the deterministic classifier."""

    model_config = ConfigDict(extra="forbid")

    rohs: NamedAliases
    reach: NamedAliases
    unsupported: list[NamedAliases]
    topics: list[AmbiguityTopic]
    regulatory_cues: list[str]


def default_catalog() -> GuardrailCatalog:
    """Return the built-in RoHS and REACH catalog."""
    return GuardrailCatalog(
        rohs=NamedAliases(
            name="RoHS",
            aliases=[
                "restriction of the use of certain hazardous substances",
                "restriction of hazardous substances",
                "directive 2011/65/eu",
                "2011/65/eu",
                "rohs directive",
                "rohs",
            ],
        ),
        reach=NamedAliases(
            name="REACH",
            aliases=[
                "registration, evaluation, authorisation and restriction of chemicals",
                "registration, evaluation, authorization and restriction of chemicals",
                "regulation (ec) no 1907/2006",
                "regulation (ec) 1907/2006",
                "(ec) no 1907/2006",
                "1907/2006",
                "reach regulation",
                "reach",
            ],
        ),
        unsupported=[
            NamedAliases(
                name="General Data Protection Regulation",
                aliases=["general data protection regulation", "regulation (eu) 2016/679", "gdpr"],
            ),
            NamedAliases(
                name="WEEE Directive",
                aliases=[
                    "waste electrical and electronic equipment directive",
                    "directive 2012/19/eu",
                    "weee directive",
                    "weee",
                ],
            ),
            NamedAliases(
                name="Low Voltage Directive",
                aliases=["low voltage directive", "directive 2014/35/eu", "lvd"],
            ),
            NamedAliases(
                name="Machinery Directive",
                aliases=["machinery directive", "directive 2006/42/ec"],
            ),
            NamedAliases(
                name="CLP Regulation",
                aliases=["classification, labelling and packaging regulation", "clp regulation", "clp"],
            ),
            NamedAliases(
                name="EMC Directive",
                aliases=["electromagnetic compatibility directive", "emc directive"],
            ),
            NamedAliases(
                name="Batteries Directive",
                aliases=["batteries directive", "directive 2006/66/ec"],
            ),
            NamedAliases(
                name="Toy Safety Directive",
                aliases=["toy safety directive", "directive 2009/48/ec"],
            ),
        ],
        topics=[
            AmbiguityTopic(
                topic="lead",
                phrases=["lead"],
                clarification="Lead is restricted under both RoHS and REACH.",
            ),
            AmbiguityTopic(
                topic="mercury",
                phrases=["mercury"],
                clarification="Mercury is restricted under both RoHS and REACH.",
            ),
            AmbiguityTopic(
                topic="cadmium",
                phrases=["cadmium"],
                clarification="Cadmium is restricted under both RoHS and REACH.",
            ),
            AmbiguityTopic(
                topic="ce_marking",
                phrases=["ce marking", "ce mark"],
                clarification=(
                    "CE marking is part of RoHS and also of other instruments, "
                    "including the Low Voltage Directive, which this assistant does not cover."
                ),
            ),
            AmbiguityTopic(
                topic="chemicals",
                phrases=["chemical substances", "chemical substance", "chemicals"],
                clarification="Rules for chemical substances can fall under RoHS or REACH.",
            ),
            AmbiguityTopic(
                topic="hazardous_substances",
                phrases=["hazardous substances", "hazardous substance", "restricted substances", "restricted substance"],
                clarification="Hazardous-substance restrictions exist in both RoHS and REACH.",
            ),
            AmbiguityTopic(
                topic="exemptions",
                phrases=["exemptions", "exemption"],
                clarification="Both RoHS and REACH contain exemptions.",
            ),
            AmbiguityTopic(
                topic="authorisation",
                phrases=["authorisation", "authorization"],
                clarification="Authorisation wording needs a named regulation before it can be answered.",
            ),
            AmbiguityTopic(
                topic="registration",
                phrases=["registration requirements", "obligation to register"],
                clarification="Registration requirements need a named regulation before they can be answered.",
            ),
            AmbiguityTopic(
                topic="market",
                phrases=["placed on the market", "placing on the market"],
                clarification="Placing on the market is used by both RoHS and REACH.",
            ),
            AmbiguityTopic(
                topic="marking",
                phrases=["marking"],
                clarification="Marking requirements can come from RoHS or from other instruments.",
            ),
        ],
        regulatory_cues=[
            "substance",
            "substances",
            "directive",
            "regulation",
            "compliance",
            "importer",
            "manufacturer",
            "mixture",
            "svhc",
            "homogeneous",
            "annex",
            "article",
        ],
    )
