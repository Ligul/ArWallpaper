if string.find(get_window_name(),"ArWallpaper [%a%d%.]+") then
	undecorate_window();
	pin_window();
	stick_window();
	set_skip_tasklist(true);
	set_skip_pager(true);
	set_window_below();
	maximize();
	set_window_type("_NET_WM_WINDOW_TYPE_DESKTOP");
end
