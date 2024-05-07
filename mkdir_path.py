def mkdir_path(path):
	import os
	isExists=os.path.exists(path)
	if not isExists:
	    os.makedirs(path)
	    return True
	else:
	    return False
