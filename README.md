# Custom-Domain-Adaptation
Imprementation of Custom Domain Adaptation (CDA) method for cross-subject EEG-based cognitive load recognition. CDA was designed

# Dependencies
	- Python (>= 3.6)
	- Tensorflow (>= 1.9)
	- NumPy (>= 1.8.2)
	- SciPy (>= 0.13.3)

# Run code

	CUDA_VISIBLE_DEVICES=0 python3 run_main.py --model resrecnet --dataset pbashivan --output outputs/resrecnet
