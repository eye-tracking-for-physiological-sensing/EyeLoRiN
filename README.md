# EyeLoRiN

- 2025 Jun: Our paper, "Inference-Time Gaze Refinement for Micro-Expression Recognition: Enhancing Event-Based Eye Tracking with Motion-Aware Post-Processing", is accepted at the [IJCAI 2025 Workshop for Micro-expression Recognition](https://jingjingchen-pro.github.io/4DMR2025/)! [Paper](https://ceur-ws.org/Vol-4115/paper3.pdf)
- 2025 Apr: Our method won second place at the [CVPR 2025 Event-based Eye Tracking Competition](https://lab-ics.github.io/3et-2025.github.io/)! [Award ceremony slides](https://docs.google.com/presentation/d/1Fbk2UOAekj5SfXke3-b2WZuArfGeaeHWwL7iFe6Rdzs/edit?slide=id.ge1065b5b8d_0_16#slide=id.ge1065b5b8d_0_16)

[Paper Link](https://arxiv.org/pdf/2506.12524)

## Abstract

Event-based eye tracking holds significant promise for fine-grained cognitive state inference, offering high temporal resolution and robustness to motion artifacts, critical features for decoding subtle mental states such as attention, confusion, or fatigue. In this work, we introduce a model-agnostic, inference-time refinement framework designed to enhance the output of existing event-based gaze estimation models without modifying their architecture or requiring retraining. Our method comprises two key post-processing modules: (i) Motion-Aware Median Filtering, which suppresses blink-induced spikes while preserving natural gaze dynamics, and (ii) Optical Flow-Based Local Refinement, which aligns gaze predictions with cumulative event motion to reduce spatial jitter and temporal discontinuities. To complement traditional spatial accuracy metrics, we propose a novel Jitter Metric that captures the temporal smoothness of predicted gaze trajectories based on velocity regularity and local signal complexity. Together, these contributions significantly improve the consistency of event-based gaze signals, making them better suited for downstream tasks such as micro-expression analysis and mind-state decoding. Our results demonstrate consistent improvements across multiple baseline models on controlled datasets, laying the groundwork for future integration with multimodal affect recognition systems in real-world environments.

## Citation

If you find our work useful and/or use in your research work, including the methods, please consider giving a star ⭐ and citing our paper.
```bibtex
@misc{bandara2025inferencetimegazerefinementmicroexpression,
      title={Inference-Time Gaze Refinement for Micro-Expression Recognition: Enhancing Event-Based Eye Tracking with Motion-Aware Post-Processing}, 
      author={Nuwan Bandara and Thivya Kandappu and Archan Misra},
      year={2025},
      eprint={2506.12524},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.12524}, 
}
```

Please contact Nuwan at pmnsbandara@smu.edu.sg if you have any issues concerning this work. 
