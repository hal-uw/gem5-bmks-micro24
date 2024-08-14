docker run -u $UID:$GID --volume $(pwd):$(pwd) -w $(pwd) pdalmia/cpcoh gem5/build/GCN3_X86/gem5.opt gem5/configs/example/apu_se.py --reg-alloc-policy=dynamic -n 3 -c bin/dwt2d --options="../data/dwt2d/192.bmp -d 192x192 -f -5 -l 3"

ls

docker run -u $UID:$GID --volume $(pwd):$(pwd) -w $(pwd) pdalmia/cpcoh gem5/build/GCN3_X86/gem5.opt gem5/configs/example/apu_se.py --reg-alloc-policy=dynamic -n 3 -c bin/dwt2d --options="../data/dwt2d/rgb.bmp -d 1024x1024 -f -5 -l 3"
