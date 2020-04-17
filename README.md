# Custom-Domain-Adaptation
Imprementation of Custom Domain Adaptation (CDA) method for cross-subject EEG-based cognitive load recognition. CDA is based on Adaptive Batch Normalization (AdaBN) and Maximimum Mean Discrepancy (MMD) to reduce the divergences between source and target distributions, respectively. Our proposal was applied on a public dataset for cross-subject cognitive load recognition. Experimental results showed that CDA obtained an accuracy of 98.2±2.67%.

# Dependencies
	
	Python (>= 3.6)
	Tensorflow (>= 1.9)
	NumPy (>= 1.8.2)
	SciPy (>= 0.13.3)

# Dataset repository
	
	https://github.com/pbashivan/EEGLearn

# Run code
	
	CUDA_VISIBLE_DEVICES=0 python3 run_main.py --model resrecnet --dataset pbashivan --output outputs/resrecnet
	
