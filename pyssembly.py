#!/usr/bin/python3
# Pyssembly

from importlib.util import MAGIC_NUMBER
from utils import *; logstart('Pyssembly')

stddef = {
	'POP': 'POP_TOP',
	'ROT': 'ROT_TWO',
	'ROT3': 'ROT_THREE',
	'DUP': 'DUP_TOP',
	'DUP2': 'DUP_TOP_TWO',

	'ADD': 'BINARY_ADD',
	'SUB': 'BINARY_SUBTRACT',
	'MUL': 'BINARY_MULTIPLY',
	'MATMUL': 'BINARY_MATRIX_MULTIPLY',
	'DIV': 'BINARY_TRUE_DIVIDE',
	'FLOORDIV': 'BINARY_FLOOR_DIVIDE',
	'MOD': 'BINARY_MODULO',
	'POW': 'BINARY_POWER',
	'LSHIFT': 'BINARY_LSHIFT',
	'RSHIFT': 'BINARY_RSHIFT',
	'AND': 'BINARY_AND',
	'OR': 'BINARY_OR',
	'XOR': 'BINARY_XOR',
	'SUBSCR': 'BINARY_SUBSCR',

	'INP_ADD': 'INPLACE_ADD',
	'INP_SUB': 'INPLACE_SUBTRACT',
	'INP_MUL': 'INPLACE_MULTIPLY',
	'INP_MATMUL': 'INPLACE_MATRIX_MULTIPLY',
	'INP_DIV': 'INPLACE_TRUE_DIVIDE',
	'INP_FLOORDIV': 'INPLACE_FLOOR_DIVIDE',
	'INP_MOD': 'INPLACE_MODULO',
	'INP_POW': 'INPLACE_POWER',
	'INP_LSHIFT': 'INPLACE_LSHIFT',
	'INP_RSHIFT': 'INPLACE_RSHIFT',
	'INP_AND': 'INPLACE_AND',
	'INP_OR': 'INPLACE_OR',
	'INP_XOR': 'INPLACE_XOR',

	'POS': 'UNARY_POSITIVE',
	'NEG': 'UNARY_NEGATIVE',
	'NOT': 'UNARY_NOT',
	'INV': 'UNARY_INVERT',

	'GETATTR': 'LOAD_ATTR',
	'SETATTR': 'STORE_ATTR',
	'DELATTR': 'DELETE_ATTR',

	'ITER': 'GET_ITER',
	'FOR': 'FOR_ITER',
	'IMPORT': 'IMPORT_NAME',
	'CALL': 'CALL_FUNCTION',
	'CALLKW': 'CALL_FUNCTION_KW',
	'CALLEX': 'CALL_FUNCTION_EX',
	'RET': 'RETURN_VALUE',

	'JUMPA': 'JUMP_ABSOLUTE',
	'JUMPF': 'JUMP_FORWARD',
	'JPOPT': 'POP_JUMP_IF_TRUE',
	'JPOPF': 'POP_JUMP_IF_FALSE',
	'JTORPOP': 'JUMP_IF_TRUE_OR_POP',
	'JFORPOP': 'JUMP_IF_FALSE_OR_POP',

	'BOOL': 'CALL_FUNCTION	(bool) 1',
	'PRINT': 'CALL_FUNCTION	(print)',
}
FLAGS = bidict.bidict(dis.COMPILER_FLAG_NAMES).inv

class PyssemblyException(Exception): pass
class PyssemblyError(PyssemblyException): pass
class PyssemblyCompileError(PyssemblyError):
	def __init__(self, *args, code, instr):
		super().__init__(*args)
		self.code, self.instr = code, instr

class Code:
	@init_defaults
	@autocast
	# TODO: @copyargs
	def __init__(self, src, *, name='<pyssembly>', filename='<string>', srclnotab=None,
		     consts: list, argnames: list, kwonlyargcount: int, varargs: bool, varkeywords: bool):
		self.name, self.filename, self.srclnotab, self.consts, self.kwonlyargcount = name, filename, srclnotab, consts, kwonlyargcount
		self.argcount = len(argnames)
		self.varnames = argnames
		self.flags = ((varargs << FLAGS['VARARGS']) |
			      (varkeywords << FLAGS['VARKEYWORDS']))
		self.instrs = list()
		self.names = list()
		self.firstlineno = 1
		self.freevars = () # TODO
		self.cellvars = () # TODO
		self.types = dict()
		self.labels = indexset()
		self.symbols = stddef.copy()

		ret = bool()
		lineno = int()
		skip = bool()
		for ii, i in enumerate(src.split('\n')):
			lineno += self.srclnotab[ii] if (self.srclnotab is not None) else +1
			if (not i.strip()): continue
			if (i[0] == '#'):
				action, _, value = map(str.strip, i.partition(' '))
				if (action == '#define'): self.symbols[value] = value
				elif (action == '#undef'): del self.symbols[value]
				elif (action == '#if'): skip = bool(eval(value))
				elif (action == '#ifdef'): skip = value not in self.symbols
				elif (action == '#ifndef'): skip = value in self.symbols
				elif (action == '#endif'): skip = False
				continue
			if (skip): continue
			opcode, *args = functools.reduce(lambda x, y: re.sub(rf"\b{y}\b", self.symbols[y], x), (i, *self.symbols)).split()
			instr = mkinstr(opcode, *args, code=self, lineno=lineno)
			if (instr is None): continue
			self.instrs.append(instr)
			if (self.stacklen < 0): raise PyssemblyCompileError(f"stack exhausted ({instr})", code=self, instr=instr)
			if (instr.opcode == dis.opmap['RETURN_VALUE']): ret = True

		if (not ret):
			self.instrs.append(mkinstr('LOAD_CONST', '(None)', code=self))
			self.instrs.append(mkinstr('RETURN_VALUE', code=self))

		#if (self.stacklen != 0): logexception(Warning(f"stack might be not empty (size={self.stacklen})")) # TODO FIXME

	def __repr__(self):
		return pformat(self.instrs)

	def to_code(self):
		return CodeType(
			self.argcount,
			self.kwonlyargcount,
			self.nlocals,
			self.stacksize,
			self.flags,
			self.codestring,
			tuple(self.consts),
			tuple(self.names),
			tuple(self.varnames),
			self.filename,
			self.name,
			self.firstlineno,
			self.lnotab,
			tuple(self.freevars),
			tuple(self.cellvars),
		)

	@property
	def nlocals(self):
		return len(self.varnames)

	@property
	def stacksize(self):
		se = int()
		ss = int()
		for i in self.instrs:
			se += i.stack_effect
			ss = max(ss, se)
		return ss

	@property
	def stacklen(self):
		return sum(i.stack_effect for i in self.instrs)

	@property
	def codestring(self):
		return bytes().join(i.pack() for i in self.instrs)

	@property
	def lnotab(self):
		r = bytearray()
		lastln = 1
		off = -2
		for i in self.instrs:
			off += 2
			if (i.lineno is None): continue
			if (i.lineno-lastln):
				r += bytes((off, i.lineno-lastln))
				off = 0
			lastln = i.lineno
		return bytes(r)

def isname(arg): return re.match(r'^\(\w+\)$', arg) is not None and arg[1:-1].isidentifier() and not keyword.iskeyword(arg[1:-1])
def islocal(arg): return re.match(r'^\$\w+$', arg) is not None
def isconst(arg): return re.match(r'^\(.+\)$', arg) is not None
def islabel(arg): return re.match(r'^:\w+$', arg) is not None
def isdir(arg): return re.match(r'^\.\w+$', arg) is not None

def mkinstr(token, *args, code, **kwargs):
	if (islabel(token)): opcode = 'Label'
	elif (isdir(token)): opcode = 'Directive'
	else: opcode = token
	return instr_types.get(opcode, Instruction).build(token, *args, code=code, **kwargs)

class Instruction:
	def __init__(self, *, opcode=None, code, lineno):
		self.opcode, self.code, self.lineno = opcode or self.__name__, code, lineno

	def __repr__(self):
		return f"{self.opcode}{f' {self.args}' if (self.args) else ''}{f' at line {self.lineno}' if (self.lineno is not None) else ''}"

	@classmethod
	def build(cls, opcode, *args, code, lineno=None):
		instr = cls(opcode=opcode, code=code, lineno=lineno)

		args = list(args)
		for ii, arg in enumerate(args):
			if (not arg and instr.hasarg): raise PyssemblyCompileError(f"Opcode {instr.opcode} needs an argument.", code=code, instr=instr)
			if (instr.hasconst):
				if (isconst(arg)):
					val = eval(arg[1:-1])
					for jj, j in enumerate(code.consts):
						if (j is val): args[ii] = jj; break
					else: args[ii] = len(code.consts); code.consts.append(val)
			elif (instr.hasname):
				if (isname(arg)):
					name = arg[1:-1]
					try: args[ii] = code.names.index(name)
					except ValueError: args[ii] = len(code.names); code.names.append(name)
			elif (instr.haslabel):
				if (islabel(arg)): args[ii] = code.labels[arg]

		instr.set_args(*args)
		return instr

	def set_args(self, *args):
		self.args = list(map(int, args))

	def pack(self):
		return bytes((self.opnum, self.arg(0)))

	def arg(self, n):
		if (not self.args): return 0
		if (self.haslabel):
			label = self.code.labels.values[self.args[n]]
			if (self.hasjabs):
				try: return S(self.code.instrs).rindex(label)*2
				except ValueError: pass
			try: return (self.code.instrs[self.code.instrs.index(self):].index(label)-1)*2
			except ValueError: pass
			raise PyssemblyCompileError(f"Label '{label}' not found", code=self.code, instr=self)
		return self.args[n]

	@property
	def opnum(self):
		return dis.opmap[self.opcode]

	@property
	def hasarg(self):
		return self.opnum >= dis.HAVE_ARGUMENT

	@property
	def hasfree(self):
		return self.opnum in dis.hasfree

	@property
	def hasjabs(self):
		return self.opnum in dis.hasjabs

	@property
	def hasjrel(self):
		return self.opnum in dis.hasjrel

	@property
	def hasname(self):
		return self.opnum in dis.hasname

	@property
	def hasconst(self):
		return self.opnum in dis.hasconst

	@property
	def haslabel(self):
		return self.hasjabs or self.hasjrel

	@property
	def haslocal(self):
		return self.opnum in dis.haslocal

	@property
	def hasnargs(self):
		return self.opnum in dis.hasnargs

	@property
	def hascompare(self):
		return self.opnum in dis.hascompare

	@property
	def stack_effect(self):
		#try: 
		return dis.stack_effect(self.opnum, 0 if (self.haslabel) else self.arg(0)) if (self.args) else dis.stack_effect(self.opnum)
		#except ValueError as ex: raise ValueError(self) from ex

	def isbinary(self):
		return self.opnum in {v for k, v in dis.opmap.items() if (k.startswith('BINARY') or k.startswith('INPLACE'))}

class Label(Instruction):
	args = []
	opcode = None
	stack_effect = 0

	def __init__(self, name, *, code, lineno):
		self.name, self.code, self.lineno = name, code, lineno

	def __eq__(self, x):
		return super() == x or self.name == x

	@classmethod
	def build(cls, name, *, code, lineno=None):
		return cls(name, code=code, lineno=lineno)

	def pack(self):
		return bytes((dis.opmap['NOP'], 0))

class Directive(Instruction):
	stack_effect = 0

	def __init__(self, dir, *args, code, lineno):
		self.dir, self.args, self.code, self.lineno = dir, args, code, lineno

	@classmethod
	def build(cls, dir, *args, code, lineno=None):
		return subclassdict(cls)[dir[1:]].build(*args, code=code, lineno=lineno)

	def pack(self):
		return b''

class print_stack(Directive):
	@classmethod
	def build(cls, *, code, lineno):
		ss = int()
		effect = list()
		for i in code.instrs:
			se = i.stack_effect
			ss += se
			if (se < 0 and len(effect) >= -se):
				effect = effect[:se]
				if (effect and i.isbinary()): effect[-1] = (i, 1)
				continue
			if (se != 0): effect.append((i, se))
		print('Stack size:', ss)
		if (effect): print('Effect:', *(Sstr(f"{i[0]}: {S(i[1]).pm()}").indent() for i in effect), sep='\n')

class print_tos(Directive):
	@classmethod
	def build(cls, *, code, lineno):
		code.instrs += [
			mkinstr('DUP_TOP', code=code),
			mkinstr('PRINT_EXPR', code=code)
		]

class ROTPOP(Instruction):
	@classmethod
	def build(cls, opcode, *args, code, lineno=None):
		code.instrs.append(mkinstr('ROT_TWO', code=code))
		return mkinstr('POP_TOP', *args, code=code, lineno=lineno)

class STORE(Instruction):
	@classmethod
	def build(cls, opcode, *args, code, lineno=None):
		if (isname(args[0])): return mkinstr('STORE_NAME', *args, code=code, lineno=lineno)
		if (islocal(args[0])): return mkinstr('STORE_FAST', *args, code=code, lineno=lineno)
		raise WTFException(args[0])

class LOAD(Instruction):
	@classmethod
	def build(cls, opcode, *args, code, lineno=None):
		if (isname(args[0])): return mkinstr('LOAD_NAME', *args, code=code, lineno=lineno)
		if (islocal(args[0])): return mkinstr('LOAD_FAST', *args, code=code, lineno=lineno)
		return mkinstr('LOAD_CONST', ' '.join(args), code=code, lineno=lineno)

class CALL_FUNCTION(Instruction):
	@autocast
	def set_args(self, f=None, argc: int = 0):
		if (isname(f)):
			self.args = [argc]
			p = len(self.code.instrs)
			se = self.stack_effect
			while (se):
				p -= 1
				se += self.code.instrs[p].stack_effect
			self.code.instrs.insert(p, mkinstr('LOAD_GLOBAL', f, code=self.code))
		elif (f is not None): self.args = [int(f)]

class CALL_FUNCTION_KW(Instruction):
	@autocast
	def set_args(self, f=None, argc: int = 0):
		if (isname(f)):
			self.args = [argc]
			p = len(self.code.instrs)
			se = self.stack_effect
			while (se):
				p -= 1
				se += self.code.instrs[p].stack_effect
			self.code.instrs.insert(p, mkinstr('LOAD_GLOBAL', f, code=self.code))
		elif (f is not None): self.args = [int(f)]

class STORE_FAST(Instruction):
	@autocast
	def set_args(self, name=None, namei: int = 0):
		if (islocal(name)):
			self.args = [len(self.code.varnames)]
			self.code.varnames.append(name[1:])
		elif (name is not None): self.args = [int(name)]

class LOAD_FAST(Instruction):
	@autocast
	def set_args(self, name=None, namei: int = 0):
		if (islocal(name)): self.args = [self.code.varnames.index(name[1:])]
		elif (name is not None): self.args = [int(name)]

instr_types = subclassdict(Instruction)

class Optimization: pass

#class 

@dispatch
def asm(src: str): return asm(Code(src))
@dispatch
def asm(code: Code): return asm(code.to_code())
@dispatch
def asm(code: CodeType):
	mtime = int(time.time())
	header = MAGIC_NUMBER
	if (sys.version_info.major == 3 and sys.version_info.minor < 7): header += struct.pack('II', mtime, 0)
	else: header += struct.pack('4xII', int(time.time()), 0)
	return header+marshal.dumps(code)

def jit(f):
	code = Code(f.__doc__, filename=inspect.getfile(f))
	code.flags |= f.__code__.co_flags
	f.__code__ = code.to_code()
	return f

def main(cargs):
	if (cargs.o is None and not cargs.file.name.rpartition('.')[0]):
		argparser.add_argument('-o', metavar='<output>', required=True)
		cargs = argparser.parse_args()
	c = Code(cargs.file.read(), filename=cargs.file.name)
	code = c.to_code()
	if (loglevel >= 1):
		dis.show_code(code)
		print('Code:')
		dis.disassemble(code)
	log("\033[92mCompilation successful.\033[0m")
	open(cargs.o or cargs.file.name.rpartition('.')[0]+'.pyc', 'wb').write(asm(c))

if (__name__ == '__main__'):
	argparser.add_argument('file', metavar='<file>', type=argparse.FileType('r'))
	argparser.add_argument('-o', metavar='<output>')
	cargs = argparser.parse_args()
	logstarted(); exit(main(cargs))
else: logimported()

# by Sdore, 2019
