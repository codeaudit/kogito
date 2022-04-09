from typing import Any
from enum import Enum
from kogito.core.head import KnowledgeHeadType


class KnowledgeRelationType(Enum):
    TRANSOMCS = "transomcs"
    ATOMIC = "atomic"
    CONCEPTNET = "conceptnet"

    def __repr__(self):
        return str(self.value)


class KnowledgeRelation:
    def __init__(self, text: str, type: KnowledgeRelationType = KnowledgeRelationType.ATOMIC) -> None:
        self.text = text
        self.type = type
    
    def verbalize(self):
        return f"Event: {self.text}"

    def __repr__(self):
        return str(self.text)


KG_RELATIONS = [
    "AtLocation",
    "CapableOf",
    "Causes",
    "CausesDesire",
    "CreatedBy",
    "DefinedAs",
    "DesireOf",
    "Desires",
    "HasA",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPainCharacter",
    "HasPainIntensity",
    "HasPrerequisite",
    "HasProperty",
    "HasSubEvent",
    "HasSubevent",
    "HinderedBy",
    "InheritsFrom",
    "InstanceOf",
    "IsA",
    "LocatedNear",
    "LocationOfAction",
    "MadeOf",
    "MadeUpOf",
    "MotivatedByGoal",
    "NotCapableOf",
    "NotDesires",
    "NotHasA",
    "NotHasProperty",
    "NotIsA",
    "NotMadeOf",
    "ObjectUse",
    "PartOf",
    "ReceivesAction",
    "RelatedTo",
    "SymbolOf",
    "UsedFor",
    "isAfter",
    "isBefore",
    "isFilledBy",
    "oEffect",
    "oReact",
    "oWant",
    "xAttr",
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xReason",
    "xWant",
]

CONCEPTNET_RELATIONS = [
    "AtLocation",
    "CapableOf",
    "Causes",
    "CausesDesire",
    "CreatedBy",
    "DefinedAs",
    "Desires",
    "HasA",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPrerequisite",
    "HasProperty",
    "HasSubevent",
    "InheritsFrom",
    "InstanceOf",
    "IsA",
    "MadeOf",
    "MotivatedByGoal",
    "NotCapableOf",
    "NotDesires",
    "NotHasA",
    "NotHasProperty",
    "NotIsA",
    "NotMadeOf",
    "PartOf",
    "ReceivesAction",
    "SymbolOf",
    "UsedFor",
]

ATOMIC_RELATIONS = [
    "AtLocation",
    "CapableOf",
    "Causes",
    "Desires",
    "HasProperty",
    "HasSubEvent",
    "HinderedBy",
    "MadeUpOf",
    "NotDesires",
    "isAfter",
    "isBefore",
    "isFilledBy",
    "oEffect",
    "oReact",
    "oWant",
    "xAttr",
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xReason",
    "xWant",
    "ObjectUse",
]

PHYSICAL_RELATIONS = [
    "ObjectUse",
    "CapableOf",
    "MadeUpOf",
    "HasProperty",
    "Desires",
    "NotDesires",
    "AtLocation",
]

EVENT_RELATIONS = [
    "Causes",
    "HinderedBy",
    "xReason",
    "isAfter",
    "isBefore",
    "HasSubEvent",
    "isFilledBy",
]

SOCIAL_RELATIONS = [
    "xIntent",
    "xReact",
    "oReact",
    "xAttr",
    "xEffect",
    "xNeed",
    "xWant",
    "oEffect",
    "oWant",
]

NOUN_PHRASE_RELATIONS = PHYSICAL_RELATIONS
SENTENCE_RELATIONS = EVENT_RELATIONS + SOCIAL_RELATIONS
VERB_PHRASE_RELATIONS = EVENT_RELATIONS

HEAD_TO_RELATION_MAP = {
    KnowledgeHeadType.SENTENCE: SENTENCE_RELATIONS,
    KnowledgeHeadType.NOUN_PHRASE: NOUN_PHRASE_RELATIONS,
    KnowledgeHeadType.VERB_PHRASE: VERB_PHRASE_RELATIONS,
}

CONCEPTNET_TO_ATOMIC_MAP = {
    "Causes":           ["Causes", "xEffect"],
    "CausesDesire":     "xWant",
    "MadeOf":           "MadeUpOf",
    "HasA":             ["MadeUpOf", "HasProperty"],
    "HasPrerequisite":  "xNeed",
    "HasSubevent":      "HasSubEvent",
    "HasFirstSubevent": "HasSubEvent",
    "HasLastSubevent":  "HasSubEvent",
    "MotivatedByGoal":  ["xIntent", "xReason"],
    "PartOf":           "MadeUpOf",
    "UsedFor":          "ObjectUse",
    "ReceivesAction":   ["MadeUpOf", "AtLocation", "Causes", "ObjectUse"]
}