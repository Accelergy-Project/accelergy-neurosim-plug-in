make:
	cp -r DNN_NeuroSim_V1.3/Inference_pytorch/NeuroSIM ./
	cp -rf drop_in/* ./NeuroSIM/
	cd NeuroSIM && $(MAKE)