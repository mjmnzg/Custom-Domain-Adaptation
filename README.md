# Custom-Domain-Adaptation
Imprementation of Custom Domain Adaptation (CDA) method for cross-subject EEG-based cognitive load recognition. CDA is based on Adaptive Batch Normalization (AdaBN) and Maximimum Mean Discrepancy (MMD) to reduce the marginal and conditional divergences between source and target distributions. Our proposal was applied on a public dataset for cross-subject cognitive load recognition. Experimental results showed that CDA obtained an accuracy of 98.2±2.67% using Leave One-Subject-Out Cross-Validation.

# Cite

	@ARTICLE{9076296,
	author={Jiménez-Guarneros, Magdiel and Gómez-Gil, Pilar},
	journal={IEEE Signal Processing Letters}, 
	title={Custom Domain Adaptation: A New Method for Cross-Subject, EEG-Based Cognitive Load Recognition}, 
	year={2020},
	volume={27},
	pages={750-754}
	}

# Dependencies
	
	Python (>= 3.6)
	Tensorflow (>= 1.9)
	NumPy (>= 1.8.2)
	SciPy (>= 0.13.3)

# Dataset repository
	
	https://github.com/pbashivan/EEGLearn

# Run code

Pre-training

	CUDA_VISIBLE_DEVICES=0 python3 run_main.py --model recresnet --dir_output model/recresnet --dir_resume outputs/resume --seed 223

Training of Custom Domain Adaptation (CDA)

	CUDA_VISIBLE_DEVICES=0 python3 run_main.py --model cda --dir_output model/cda --dir_pretrain model/recresnet --dir_resume outputs/resume --seed 223
