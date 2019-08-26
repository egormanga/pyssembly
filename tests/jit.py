import pyssembly

@pyssembly.jit
def test(): '''
	LOAD	(2)
	LOAD	(3)
	DIV
	CALL	(print) 1
'''

test()
