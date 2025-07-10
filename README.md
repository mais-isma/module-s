# Module S - Symbolic Ethical World Model

**Version**: 1.0  
**Author**: Ismaël Martin Fernandez  
**License**: Open Source (MIT)

---

## What is Module S?

**Module S** is the first complete symbolic layer for AGI (Artificial General Intelligence), integrating ethical decision-making through a semantic memory network guided by **7 universal virtues**:

> Wisdom, Compassion, Mercy, Righteousness, Love, Humility, and Charity

Combined with negative anchors such as Danger, Chaos, and Error, this model enables symbolic abstraction, moral evaluation, and generalization across tasks.

It is implemented on reinforcement learning agents to solve the **catastrophic forgetting problem**, and to create truly interpretable, ethical, and value-aligned AI systems.

---

## Key Features

- **Symbolic Graph Layer**: maps states to concepts and virtues
- **Concept Dynamics**: forward prediction of semantic consequences of actions
- **Moral Weighting**: decisions guided by virtue-weighted concept probabilities
- **Memory Regulation**: keeps and removes knowledge based on ethical relevance
- **Semantic Planning**: forward model uses concepts to plan actions over horizons
- **Multi-task Adaptation**: symbolic retention allows adaptation to new tasks without forgetting

---

## How to Use It

1. Clone the repository
```bash
git clone https://github.com/mais-isma/module-s.git
cd module-s
```

2. Install requirements
```bash
pip install -r requirements.txt
```

3. Run the main script on LunarLander-v2 (Gym)
```bash
python module_s.py
```

4. Visualize the symbolic graph
```bash
# After training, an image will be saved: Task_1_graph.png
```

---

##  Files Structure

- `module_s.py` : main symbolic learning script
- `README.md` : documentation and project info
- `graphs/` : semantic graphs generated after training
- `logs/` : optional logging output

---

##  Academic Resources

- [Academia.edu Publication](https://independent.academia.edu/IsmaelFernandez103) 
---

##  Contact / Publication

You can cite this work as:
```
Ismaël Martin Fernandez. Module S: Symbolic Ethical Learning for General AI. 2025.
```

No collaboration is requested. This release is for **academic dissemination only**. The author does not wish to co-develop the model further but grants open usage.

Contact: [Ismaël Martin Fernandez](mailto:maisldv@gmail.com)

---

