#!/usr/bin/python3
# Pyssembly

from importlib.util import MAGIC_NUMBER
from utils import *; logstart('Pyssembly')

stddef = {
	'POP': 'POP_TOP',
	'ROT': 'ROT_TWO',
	'ROT2': 'ROT_TWO',
	'ROT3': 'ROT_THREE',
	'DUP': 'DUP_TOP',
	'DUP2': 'DUP_TOP_TWO',

	'ADD': 'BINARY_ADD',
	'SUB': 'BINARY_SUBTRACT',
	'MUL': 'BINARY_MULTIPLY',
	'MMUL': 'BINARY_MATRIX_MULTIPLY',
	'MATMUL': 'BINARY_MATRIX_MULTIPLY',
	'DIV': 'BINARY_TRUE_DIVIDE',
	'IDIV': 'BINARY_FLOOR_DIVIDE',
	'FLOORDIV': 'BINARY_FLOOR_DIVIDE',
	'MOD': 'BINARY_MODULO',
	'POW': 'BINARY_POWER',
	'LSH': 'BINARY_LSHIFT',
	'SHL': 'BINARY_LSHIFT',
	'LSHIFT': 'BINARY_LSHIFT',
	'RSH': 'BINARY_RSHIFT',
	'SHR': 'BINARY_RSHIFT',
	'RSHIFT': 'BINARY_RSHIFT',
	'AND': 'BINARY_AND',
	'XOR': 'BINARY_XOR',
	'OR': 'BINARY_OR',
	'SUBSCR': 'BINARY_SUBSCR',

	'IPADD': 'INPLACE_ADD',
	'INP_ADD': 'INPLACE_ADD',
	'IPSUB': 'INPLACE_SUBTRACT',
	'INP_SUB': 'INPLACE_SUBTRACT',
	'IPMUL': 'INPLACE_MULTIPLY',
	'INP_MUL': 'INPLACE_MULTIPLY',
	'IPMMUL': 'INPLACE_MATRIX_MULTIPLY',
	'IPMATMUL': 'INPLACE_MATRIX_MULTIPLY',
	'INP_MATMUL': 'INPLACE_MATRIX_MULTIPLY',
	'IPDIV': 'INPLACE_TRUE_DIVIDE',
	'INP_DIV': 'INPLACE_TRUE_DIVIDE',
	'IPIDIV': 'INPLACE_FLOOR_DIVIDE',
	'IPFLOORDIV': 'INPLACE_FLOOR_DIVIDE',
	'INP_FLOORDIV': 'INPLACE_FLOOR_DIVIDE',
	'IPMOD': 'INPLACE_MODULO',
	'INP_MOD': 'INPLACE_MODULO',
	'IPPOW': 'INPLACE_POWER',
	'INP_POW': 'INPLACE_POWER',
	'IPLSH': 'INPLACE_LSHIFT',
	'IPSHL': 'INPLACE_LSHIFT',
	'IPLSHIFT': 'INPLACE_LSHIFT',
	'INP_LSHIFT': 'INPLACE_LSHIFT',
	'IPRSH': 'INPLACE_RSHIFT',
	'IPSHR': 'INPLACE_RSHIFT',
	'IPRSHIFT': 'INPLACE_RSHIFT',
	'INP_RSHIFT': 'INPLACE_RSHIFT',
	'IPAND': 'INPLACE_AND',
	'INP_AND': 'INPLACE_AND',
	'IPXOR': 'INPLACE_XOR',
	'INP_XOR': 'INPLACE_XOR',
	'IPOR': 'INPLACE_OR',
	'INP_OR': 'INPLACE_OR',

	'POS': 'UNARY_POSITIVE',
	'NEG': 'UNARY_NEGATIVE',
	'NOT': 'UNARY_NOT',
	'INV': 'UNARY_INVERT',

	'GETATTR': 'LOAD_ATTR',
	'SETATTR': 'STORE_ATTR',
	'DELATTR': 'DELETE_ATTR',

	'ITER': 'GET_ITER',
	'FOR': 'FOR_ITER',
	'CMP': 'COMPARE_OP',
	'IMPORT': 'IMPORT_NAME',
	'IMPFROM': 'IMPORT_FROM',
	'IMPALL': 'IMPORT_STAR',
	'LOOP': 'SETUP_LOOP', # TODO move to correct place [1]
	'UNPACK': 'UNPACK_SEQUENCE',			 # [2]
	'UNPEX': 'UNPACK_EX',				 # [3]
	'CALL': 'CALL_FUNCTION',
	'CALLKW': 'CALL_FUNCTION_KW',
	'CALLEX': 'CALL_FUNCTION_EX',
	'MKFUNC': 'MAKE_FUNCTION',
	'RET': 'RETURN_VALUE',

	'JA': 'JUMP_ABSOLUTE',
	'JUMPA': 'JUMP_ABSOLUTE',
	'JF': 'JUMP_FORWARD',
	'JUMPF': 'JUMP_FORWARD',
	'JPT': 'POP_JUMP_IF_TRUE',
	'JTP': 'POP_JUMP_IF_TRUE',
	'JPOPT': 'POP_JUMP_IF_TRUE',
	'PJIT': 'POP_JUMP_IF_TRUE',
	'JPF': 'POP_JUMP_IF_FALSE',
	'JFP': 'POP_JUMP_IF_FALSE',
	'JPOPF': 'POP_JUMP_IF_FALSE',
	'PJIF': 'POP_JUMP_IF_FALSE',
	'JTOP': 'JUMP_IF_TRUE_OR_POP',
	'JTORPOP': 'JUMP_IF_TRUE_OR_POP',
	'JFOP': 'JUMP_IF_FALSE_OR_POP',
	'JFORPOP': 'JUMP_IF_FALSE_OR_POP',

	'EXTARG': 'EXTENDED_ARG',

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
	def __init__(self, src, *, name='<pyssembly>', filename='<string>', srclnotab=None, firstlineno=None,
		     consts: list, argnames: list, posonlyargcount: int, kwonlyargcount: int, varargs: bool, varkeywords: bool):
		self.name, self.filename, self.consts, self.posonlyargcount, self.kwonlyargcount = name, filename, consts, posonlyargcount, kwonlyargcount
		self.argcount = len(argnames)
		self.varnames = argnames
		self.flags = ((varargs << FLAGS['VARARGS']) |
			      (varkeywords << FLAGS['VARKEYWORDS']))
		self.instrs = list()
		self.names = indexset()
		self.firstlineno = firstlineno if (firstlineno is not None) else srclnotab[0] if (srclnotab) else 1
		self.freevars = () # TODO
		self.cellvars = () # TODO
		self.types = dict()
		self.labels = indexset()
		self.symbols = stddef.copy()

		ret = bool()
		lineno = int()
		skip = bool()
		for ii, i in enumerate(src.split('\n')):
			lineno += (srclnotab[ii] if (ii < len(srclnotab)) else 0) if (srclnotab is not None) else +1
			i = i.strip()
			if (not i): continue
			if (i[0] == '#'):
				action, _, value = map(str.strip, i.partition(' '))
				if (action == '#endif'): skip = False
				elif (skip): continue
				elif (action == '#if'): skip = bool(eval(value))
				elif (action == '#ifdef'): skip = value not in self.symbols
				elif (action == '#ifndef'): skip = value in self.symbols
				elif (action == '#define'): self.symbols[value] = value
				elif (action == '#undef'): del self.symbols[value]
				elif (action == '#line'): lineno, srclnotab = int(value)-self.firstlineno, ()
				continue
			if (skip): continue
			opcode, *args = re.split(r'\t+', functools.reduce(lambda x, y: re.sub(rf"\b{y}\b", self.symbols[y], x), (i, *self.symbols)))
			instr = mkinstr(opcode, *args, code=self, lineno=lineno)
			if (instr is None): continue
			self.instrs.append(instr)
			#if (self.stacklen < 0): raise PyssemblyCompileError(f"stack exhausted ({instr})", code=self, instr=instr) # TODO FIXME stack_effect & print_stack
			if (instr.opnum == dis.opmap['RETURN_VALUE']): ret = True

		if (not ret):
			self.instrs.append(mkinstr('LOAD_CONST', '(None)', code=self))
			self.instrs.append(mkinstr('RETURN_VALUE', code=self))

		#if (self.stacklen != 0): logexception(Warning(f"stack might be not empty (size={self.stacklen})")) # TODO FIXME

	def __repr__(self):
		return pformat(self.instrs)

	def to_code(self):
		self.instrs = tuple(self.instrs)
		try: return CodeType(
			int(self.argcount),
			*(int(self.posonlyargcount),)*(sys.version_info.major == 3 and sys.version_info.minor >= 8),
			int(self.kwonlyargcount),
			int(self.nlocals),
			int(self.stacksize),
			int(self.flags),
			bytes(self.codestring),
			tuple(self.consts),
			tuple(self.names),
			tuple(self.varnames),
			str(self.filename),
			str(self.name),
			int(self.firstlineno),
			bytes(self.lnotab),
			tuple(self.freevars),
			tuple(self.cellvars),
		)
		except AttributeError as ex:
			if (re.fullmatch(r"'tuple' object has no attribute '(append|insert|remove|pop)'", ex.args[0])): # TODO: subclass tuple for this
				raise PyssemblyError('Cannot modify instrs during assembly.') from ex
			else: raise
		finally: self.instrs = list(self.instrs)

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
		lastln = 0
		off = int()
		for ii, i in enumerate(self.instrs):
			if (i.lineno is not None):
				if (i.lineno-lastln):
					r += bytes((off, i.lineno-lastln))
					off = 0
				lastln = i.lineno
			off += i.size
		return bytes(r)

def isname(arg): return arg is not None and (re.fullmatch(r'<.+>', arg) is not None or re.fullmatch(r'\(.+\)', arg) is not None and arg[1:-1].replace('.', '').isidentifier()) and not keyword.iskeyword(arg[1:-1])
def islocal(arg): return arg is not None and re.fullmatch(r'\$.+', arg) is not None
def isconst(arg): return arg is not None and re.fullmatch(r'\(.+\)', arg) is not None
def islabel(arg): return arg is not None and re.fullmatch(r':\w+', arg) is not None
def isdir(arg): return arg is not None and re.fullmatch(r'\.\w+', arg) is not None
def iscmpop(arg): return arg is not None and re.fullmatch(r'\(.+\)', arg) is not None and arg[1:-1] in dis.cmp_op

def mkinstr(token, *args, code, **kwargs):
	if (islabel(token)): opcode = 'Label'
	elif (isdir(token)): opcode = 'Directive'
	else: opcode = token
	return instr_types.get(opcode, Instruction).build(token, *args, code=code, **kwargs)

class Instruction:
	size = 2

	def __init__(self, *, opcode=None, code, lineno):
		self.opcode, self.code, self.lineno = opcode or self.__class__.__name__, code, lineno
		self.args = list()

	def __repr__(self):
		return f"{self.opcode}{f' {self.args}' if (self.args) else ''}{f' at line {self.lineno}' if (self.lineno is not None) else ''}"

	@classmethod
	def build(cls, opcode, *args, code, lineno=None):
		instr = cls(opcode=opcode, code=code, lineno=lineno)

		args = list(args)
		for ii, arg in enumerate(args):
			if (not arg and instr.hasarg): raise PyssemblyCompileError(f"Opcode {instr.opcode} needs an argument: {instr.usage}.", code=code, instr=instr)
			if (instr.hasconst):
				if (isconst(arg)):
					val = eval(arg[1:-1])
					for jj, j in enumerate(code.consts):
						if (j is val): args[ii] = jj; break
					else: args[ii] = len(code.consts); code.consts.append(val)
			elif (instr.hasname):
				if (isname(arg)): args[ii] = code.names[arg[1:-1]]
			#elif (instr.haslabel):
			#	if (islabel(arg)): args[ii] = code.labels[arg]
			elif (instr.hascompare):
				if (iscmpop(arg)): args[ii] = dis.cmp_op.index(arg[1:-1])

		try: instr.set_args(*args)
		except Exception as ex: raise SyntaxError(f"\t{S(' ').join((opcode, *args))}\nExpected:    \t{instr.usage}\n"+f"(on line {lineno})"*(lineno is not None)) from ex

		return instr

	def set_args(self, *args):
		self.args = list(args)#list(map(int, args))

	def pack(self):
		return bytes((self.opnum, self.arg(0)))

	def arg(self, n):
		if (not self.args): return 0
		if (self.haslabel and isinstance(self.args[n], str) and islabel(self.args[n])):
			label = self.args[n]#self.code.labels.values[self.args[n]]
			if (self.hasjabs):
				try: return sum(i.size for i in self.code.instrs[:S(self.code.instrs).rindex(label)])
				except ValueError: pass
			try: return (self.code.instrs[self.code.instrs.index(self):].index(label)-1)*2
			except ValueError: pass
			raise PyssemblyCompileError(f"Label '{label}' not found", code=self.code, instr=self)
		return int(self.args[n])

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

	@property
	def usage(self):
		return f"{self.opcode} {self.__doc__.strip() if (self.__doc__ is not None) else ' '.join(('...' if (i.kind is i.VAR_POSITIONAL) else i.name).join('<>' if (i.default is inspect._empty and i.kind is not i.VAR_POSITIONAL) else '[]') for i in inspect.signature(self.set_args if (self.set_args is not Instruction.set_args) else self.build).parameters.values() if i.name not in ('self', 'opcode', 'code', 'lineno'))}".strip()

	def isbinary(self):
		return self.opnum in {v for k, v in dis.opmap.items() if (k.startswith('BINARY') or k.startswith('INPLACE'))}

class Label(Instruction):
	''' :<label> '''

	args = []
	opcode = None
	opnum = None
	size = 0
	stack_effect = 0

	def __init__(self, name, *, code, lineno):
		self.name, self.code, self.lineno = name, code, lineno

	def __repr__(self):
		return f"{self.name}{f' at line {self.lineno}' if (self.lineno is not None) else ''}"

	def __eq__(self, x):
		return self.name == x

	@classmethod
	def build(cls, name, *, code, lineno=None):
		return cls(name, code=code, lineno=lineno)

	def pack(self):
		return b''

class Directive(Instruction):
	''' .<directive> '''

	size = 0
	stack_effect = 0

	def __init__(self, dir, *args, code, lineno):
		self.dir, self.args, self.code, self.lineno = dir, args, code, lineno

	@classmethod
	def build(cls, dir, *args, code, lineno=None):
		return subclassdict(cls)[dir[1:]].build(*args, code=code, lineno=lineno)

	def pack(self):
		return b''

class print_stack(Directive): # TODO FIXME fold ops correctly
	@classmethod
	def build(cls, *, code, lineno):
		ss = int()
		effect = list()
		for i in code.instrs:
			se = i.stack_effect
			ss += se
			if (se < 0 and len(effect) >= -se):
				effect = effect[:se]
				#if (effect and i.isbinary()): effect[-1] = (i, 1) # TODO FIXME?
				continue
			#if (se != 0): 
			effect.append((i, se))
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
	def build(cls, opcode, code, lineno=None):
		code.instrs.append(mkinstr('ROT_TWO', code=code))
		return mkinstr('POP_TOP', code=code, lineno=lineno)

class LOAD(Instruction):
	''' <name | value> '''

	@classmethod
	def build(cls, opcode, *args, code, lineno=None):
		if (isname(args[0])): return mkinstr('LOAD_NAME', *args, code=code, lineno=lineno)
		if (islocal(args[0])): return mkinstr('LOAD_FAST', *args, code=code, lineno=lineno)
		return mkinstr('LOAD_CONST', ' '.join(args), code=code, lineno=lineno)

class STORE(Instruction):
	''' [value] <name> '''

	@classmethod
	def build(cls, opcode, val, name=None, *, code, lineno=None):
		if (name is not None): code.instrs.append(mkinstr('LOAD', val, code=code, lineno=lineno))
		else: name = val
		if (isname(name)): return mkinstr('STORE_NAME', name, code=code, lineno=lineno)
		if (islocal(name)): return mkinstr('STORE_FAST', name, code=code, lineno=lineno)
		raise WTFException(name)

class DELETE(Instruction):
	@classmethod
	def build(cls, opcode, name, code, lineno=None):
		if (isname(name)): return mkinstr('DELETE_NAME', name, code=code, lineno=lineno)
		if (islocal(name)): return mkinstr('DELETE_FAST', name, code=code, lineno=lineno)
		raise WTFException(name)

class CALL_FUNCTION(Instruction):
	''' [[callable] argc] '''

	@autocast
	def set_args(self, f=None, argc: int = 0):
		if (isname(f)):
			self.args = [argc]
			p = len(self.code.instrs)
			se = self.stack_effect
			while (se):
				p -= 1
				se += self.code.instrs[p].stack_effect
			self.code.instrs.insert(p, mkinstr('LOAD_GLOBAL', f, code=self.code, lineno=self.code.instrs[p].lineno))
		elif (f is not None): self.args = [int(f)]
		else: self.args = [argc]

class CALL_FUNCTION_KW(Instruction):
	''' [[callable] argc] '''

	@autocast
	def set_args(self, f=None, argc: int = 0):
		if (isname(f)):
			self.args = [argc]
			p = len(self.code.instrs)
			se = self.stack_effect
			while (se):
				p -= 1
				se += self.code.instrs[p].stack_effect
			self.code.instrs.insert(p, mkinstr('LOAD_GLOBAL', f, code=self.code, lineno=self.code.instrs[p].lineno))
		elif (f is not None): self.args = [int(f)]
		else: self.args = [argc]

class STORE_FAST(Instruction):
	def set_args(self, var):
		if (islocal(var)):
			self.args = [len(self.code.varnames)]
			self.code.varnames.append(var[1:])
		else: self.args = [int(var)]

class LOAD_FAST(Instruction):
	def set_args(self, var):
		if (islocal(var)): self.args = [self.code.varnames.index(var[1:])]
		else: self.args = [int(var)]

class RETURN_VALUE(Instruction):
	def set_args(self, value=None):
		if (value is not None): self.code.instrs.append(mkinstr('LOAD', value, code=self.code))

class COMPARE_OP(Instruction):
	''' <operator> '''

	@classmethod
	def build(cls, opcode, *args, code, lineno=None):
		return super().build(opcode, ' '.join(args), code=code, lineno=lineno)

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
	header = MAGIC_NUMBER + struct.pack('4x'*(sys.version_info.major == 3 and sys.version_info.minor >= 7)+'II', mtime, 0)
	return header+marshal.dumps(code)

def jit(f):
	code = Code(f.__doc__, filename=inspect.getfile(f), argnames=inspect.signature(f).parameters.keys(), firstlineno=f.__code__.co_firstlineno)
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

# by Sdore, 2020
