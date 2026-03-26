import yaml
from typing import Optional
 
 
def template_normal(description: str, category: Optional[str] = None) -> str:

    data = {"Normal Scene": [description]}
    return yaml.dump(data, default_flow_style=False, allow_unicode=True, indent=4)
 
 
def template_anomalous(description: str, category: str) -> str:
    data = {
        "Anomalous Scene": {
            "Action Category": category,
            "Scene Description": description,
        }
    }
    return yaml.dump(data, default_flow_style=False, allow_unicode=True, indent=4)
 
 
def apply_repulsive_prompting(
    captions_normal:     list[str],
    captions_anomalous:  list[str],
    categories_normal:   list[str],
    categories_anomalous: list[str],
) -> tuple[list[str], list[str]]:
    assert len(captions_normal) == len(categories_normal), \
        "Length mismatch: captions_normal vs categories_normal"
    assert len(captions_anomalous) == len(categories_anomalous), \
        "Length mismatch: captions_anomalous vs categories_anomalous"
 
    templated_normal = [
        template_normal(desc, cat)
        for desc, cat in zip(captions_normal, categories_normal)
    ]
 
    templated_anomalous = [
        template_anomalous(desc, cat)
        for desc, cat in zip(captions_anomalous, categories_anomalous)
    ]
 
    return templated_normal, templated_anomalous
 
 
def apply_repulsive_prompting_concatenated(
    captions:   list[str],
    categories: list[str],
    labels:     list[int],
) -> list[str]:

    assert len(captions) == len(categories) == len(labels), \
        "Lengths of captions, categories, and labels must match."
 
    templated = []
    for desc, cat, label in zip(captions, categories, labels):
        if label == 0:
            templated.append(template_normal(desc, cat))
        else:
            templated.append(template_anomalous(desc, cat))
    return templated
 
if __name__ == "__main__":
    normal_examples = [
        ("Audiences are clapping and cheering at a public talent show event.", "Normal"),
        ("A technician repairs phones at a retail electronics store.", "Normal"),
    ]
    anomalous_examples = [
        ("A bystander attacking a newscaster during a live broadcast.", "Assault"),
        ("A thief grabs a customer's bag left on the floor and dashes out.", "Theft"),
    ]
 
    print("=" * 60)
    print("Repulsive Prompting (RP) — Template Examples")
    print("=" * 60)
 
    print("\n--- Normal Template TN (keyword: 'Normal') ---")
    for desc, cat in normal_examples:
        print(template_normal(desc, cat))
        print("-" * 40)
 
    print("\n--- Anomalous Template TA (keyword: 'Anomalous') ---")
    for desc, cat in anomalous_examples:
        print(template_anomalous(desc, cat))
        print("-" * 40)
 
    # Batch application
    cn   = [d for d, _ in normal_examples]
    kn   = [c for _, c in normal_examples]
    ca   = [d for d, _ in anomalous_examples]
    ka   = [c for _, c in anomalous_examples]
 
    tn_out, ta_out = apply_repulsive_prompting(cn, ca, kn, ka)
    print(f"\nBatch RP: {len(tn_out)} normal, {len(ta_out)} anomalous templates generated.")
 
    # Centroid angle check reminder
    print(
        "\nNote: RP widens the centroid angle between ZN and ZA "
        "from 8.12° (no RP) to 33.29° (with RP), per Table 3b of the paper."
    )

