all:
	g++ Main.cpp -g -O2 \
		-o ArWallpaper \
		-lGLEW -lX11 -lGLU -lGL \
		-lglfw \
		-lglut \
		-I/usr/include/opencv4 \
		-lopencv_gapi -lopencv_stitching -lopencv_alphamat -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_cvv -lopencv_dnn_objdetect -lopencv_dnn_superres -lopencv_dpm -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_intensity_transform -lopencv_line_descriptor -lopencv_mcc -lopencv_quality -lopencv_rapid -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_superres -lopencv_optflow -lopencv_surface_matching -lopencv_tracking -lopencv_highgui -lopencv_datasets -lopencv_text -lopencv_plot -lopencv_videostab -lopencv_videoio -lopencv_viz -lopencv_xfeatures2d -lopencv_shape -lopencv_ml -lopencv_ximgproc -lopencv_video -lopencv_dnn -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core

install:
	install -D -t /opt/ArWallpaper ArWallpaper
	install -D -t /opt/ArWallpaper launch.sh
	cp -r data /opt/ArWallpaper/
	ln -s /opt/ArWallpaper/ArWallpaper /usr/local/bin/
	cp ArWallpaper.desktop /usr/share/applications/

uninstall:
	rm -rf /opt/ArWallpaper
	rm /usr/local/bin/ArWallpaper
	rm /usr/share/applications/ArWallpaper.desktop
