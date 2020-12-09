import attr


SUPPORTED_ATTRIBUTES = [
    "simulation",
    "last_day",
    # "contact_models",
    # "policies",
]


@attr.s
class InferenceData:

    @classmethod
    def from_sid(cls):
        pass
