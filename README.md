# ML-Enhanced Guillotine Packing Algorithm

This repository contains a machine learning-based enhancement of the **Priority Heuristic (PH)** algorithm for the **Guillotine Rectangular Packing Problem (GRPP)**, originally proposed in:

> Zhang, D., Shi, L., Leung, S.C.H., & Wu, T. (2016).  
> *A priority heuristic for the guillotine rectangular packing problem*.  
> Information Processing Letters, 116(1), 15–21.  
> [DOI: 10.1016/j.ipl.2015.08.008](https://doi.org/10.1016/j.ipl.2015.08.008)

---

## 📚 Background

The original PH algorithm recursively selects and places rectangular items in a bin using a priority-based heuristic that respects guillotine cutting constraints. This algorithm was covered in my **Master's-level Algorithm Design course** at Xiamen University in 2016, taught by Prof. Defu Zhang — one of the authors of the original paper.

At that time, implementing the heuristic in code was part of the course assignments.

---

## 🔬 Project Objective

This project revisits the classical GRPP formulation and **augments it using modern machine learning techniques**, particularly:

- Predictive models for item placement (position and rotation)
- Feasibility-aware decision learning
- Enhanced visualization of packing layouts

The goal is to explore **data-driven strategies** that improve or complement traditional heuristic decisions, enabling intelligent generalization across varying item distributions.


# Acknowledgment
This project is a continuation of the algorithmic ideas taught in the 2016 graduate course "Design and Analysis of Algorithms" at Xiamen University, taught by Prof. Defu Zhang. His contributions and foundational work in GRPP inspired this modernized extension.



@article{zhang2016priority,
  title={A priority heuristic for the guillotine rectangular packing problem},
  author={Zhang, Defu and Shi, Leyuan and Leung, Stephen CH and Wu, Tao},
  journal={Information Processing Letters},
  volume={116},
  number={1},
  pages={15--21},
  year={2016},
  publisher={Elsevier}
}
