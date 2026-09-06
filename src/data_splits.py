"""Subject-level split shared with visualization."""
DEV_USERS = list("ABCDEFKL")
TEST_USERS = list("GHIJ")
FOLDS = [("val_AB", ["A", "B"]), ("val_CE", ["C", "E"]),
         ("val_DF", ["D", "F"]), ("val_KL", ["K", "L"])]

def split_plan():
    return {
        "folds": [dict(name=name, train=[u for u in DEV_USERS if u not in val],
                       validation=val[:]) for name, val in FOLDS],
        "final_train": DEV_USERS[:], "final_test": TEST_USERS[:],
        "inference": "uwb only; ground truth is used only after channel selection"
    }
