# cdips

Data in training.bin includes

Load with:
`training = pd.read_msgpack('training.bin’)`

Columns:
       'subject', 'img', 'pixels', 
	'maskArea', # number of pixels
	’maskC', # center of mask
	’maskS', # singular values
	’maskV', # rotation matrix
       'maskContour' # outline of maskArea