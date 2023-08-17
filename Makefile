make:
	cp -r DNN_NeuroSim_V1.3/Inference_pytorch/NeuroSIM ./NeuroSim
	cp -rf drop_in/* ./NeuroSim/
	cd NeuroSim && $(MAKE)


install:
	make
	pip install .
