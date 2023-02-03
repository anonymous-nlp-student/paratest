from termcolor import colored

VALID = 1
INVALID = 0


def annotate(sample, T=None, clf=None):
    if sample is None:
        return INVALID, 1
    else:
        if clf is not None:
            validity, prob = clf.predict([sample])
            return validity.item(), prob.item()

        else:
            if isinstance(sample, str):
                demo = "\n\n".join(T)
                s = sample
            else:
                if any(ss is None for ss in sample):
                    return INVALID, 1
                else:
                    demo = "\n\n".join(["\n".join(t) for t in T])
                    s = "\n".join(sample)

            capture_key = input(
                "Examples:\n{}\n\nOutput:\n{}\n\nValid: Tab, Invalid: Others\n>> ".format(
                colored(demo, "green"), s)
            )

            if capture_key == "\t":
                return VALID, 1
            else:
                return INVALID, 1