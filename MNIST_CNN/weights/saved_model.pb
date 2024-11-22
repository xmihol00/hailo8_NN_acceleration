��+
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
s
FakeQuantWithMinMaxVars

inputs
min
max
outputs"
num_bitsint"
narrow_rangebool( 
}
!FakeQuantWithMinMaxVarsPerChannel

inputs
min
max
outputs"
num_bitsint"
narrow_rangebool( 
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��$
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
|
Adam/v/dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/v/dense1/bias
u
&Adam/v/dense1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense1/bias*
_output_shapes
:
*
dtype0
|
Adam/m/dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/m/dense1/bias
u
&Adam/m/dense1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense1/bias*
_output_shapes
:
*
dtype0
�
Adam/v/dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*%
shared_nameAdam/v/dense1/kernel
~
(Adam/v/dense1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense1/kernel*
_output_shapes
:	�
*
dtype0
�
Adam/m/dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*%
shared_nameAdam/m/dense1/kernel
~
(Adam/m/dense1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense1/kernel*
_output_shapes
:	�
*
dtype0
v
Adam/v/bn3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/v/bn3/beta
o
#Adam/v/bn3/beta/Read/ReadVariableOpReadVariableOpAdam/v/bn3/beta*
_output_shapes
:*
dtype0
v
Adam/m/bn3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/m/bn3/beta
o
#Adam/m/bn3/beta/Read/ReadVariableOpReadVariableOpAdam/m/bn3/beta*
_output_shapes
:*
dtype0
x
Adam/v/bn3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/v/bn3/gamma
q
$Adam/v/bn3/gamma/Read/ReadVariableOpReadVariableOpAdam/v/bn3/gamma*
_output_shapes
:*
dtype0
x
Adam/m/bn3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/m/bn3/gamma
q
$Adam/m/bn3/gamma/Read/ReadVariableOpReadVariableOpAdam/m/bn3/gamma*
_output_shapes
:*
dtype0
z
Adam/v/conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/v/conv3/bias
s
%Adam/v/conv3/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3/bias*
_output_shapes
:*
dtype0
z
Adam/m/conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/m/conv3/bias
s
%Adam/m/conv3/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3/bias*
_output_shapes
:*
dtype0
�
Adam/v/conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/conv3/kernel
�
'Adam/v/conv3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3/kernel*&
_output_shapes
:*
dtype0
�
Adam/m/conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/conv3/kernel
�
'Adam/m/conv3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3/kernel*&
_output_shapes
:*
dtype0
v
Adam/v/bn2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/v/bn2/beta
o
#Adam/v/bn2/beta/Read/ReadVariableOpReadVariableOpAdam/v/bn2/beta*
_output_shapes
:*
dtype0
v
Adam/m/bn2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/m/bn2/beta
o
#Adam/m/bn2/beta/Read/ReadVariableOpReadVariableOpAdam/m/bn2/beta*
_output_shapes
:*
dtype0
x
Adam/v/bn2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/v/bn2/gamma
q
$Adam/v/bn2/gamma/Read/ReadVariableOpReadVariableOpAdam/v/bn2/gamma*
_output_shapes
:*
dtype0
x
Adam/m/bn2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/m/bn2/gamma
q
$Adam/m/bn2/gamma/Read/ReadVariableOpReadVariableOpAdam/m/bn2/gamma*
_output_shapes
:*
dtype0
z
Adam/v/conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/v/conv2/bias
s
%Adam/v/conv2/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2/bias*
_output_shapes
:*
dtype0
z
Adam/m/conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/m/conv2/bias
s
%Adam/m/conv2/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2/bias*
_output_shapes
:*
dtype0
�
Adam/v/conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/conv2/kernel
�
'Adam/v/conv2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2/kernel*&
_output_shapes
:*
dtype0
�
Adam/m/conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/conv2/kernel
�
'Adam/m/conv2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2/kernel*&
_output_shapes
:*
dtype0
v
Adam/v/bn1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/v/bn1/beta
o
#Adam/v/bn1/beta/Read/ReadVariableOpReadVariableOpAdam/v/bn1/beta*
_output_shapes
:*
dtype0
v
Adam/m/bn1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/m/bn1/beta
o
#Adam/m/bn1/beta/Read/ReadVariableOpReadVariableOpAdam/m/bn1/beta*
_output_shapes
:*
dtype0
x
Adam/v/bn1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/v/bn1/gamma
q
$Adam/v/bn1/gamma/Read/ReadVariableOpReadVariableOpAdam/v/bn1/gamma*
_output_shapes
:*
dtype0
x
Adam/m/bn1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/m/bn1/gamma
q
$Adam/m/bn1/gamma/Read/ReadVariableOpReadVariableOpAdam/m/bn1/gamma*
_output_shapes
:*
dtype0
z
Adam/v/conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/v/conv1/bias
s
%Adam/v/conv1/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv1/bias*
_output_shapes
:*
dtype0
z
Adam/m/conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/m/conv1/bias
s
%Adam/m/conv1/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv1/bias*
_output_shapes
:*
dtype0
�
Adam/v/conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/conv1/kernel
�
'Adam/v/conv1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv1/kernel*&
_output_shapes
:*
dtype0
�
Adam/m/conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/conv1/kernel
�
'Adam/m/conv1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv1/kernel*&
_output_shapes
:*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
n
dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense1/bias
g
dense1/bias/Read/ReadVariableOpReadVariableOpdense1/bias*
_output_shapes
:
*
dtype0
w
dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*
shared_namedense1/kernel
p
!dense1/kernel/Read/ReadVariableOpReadVariableOpdense1/kernel*
_output_shapes
:	�
*
dtype0
~
bn3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namebn3/moving_variance
w
'bn3/moving_variance/Read/ReadVariableOpReadVariableOpbn3/moving_variance*
_output_shapes
:*
dtype0
v
bn3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namebn3/moving_mean
o
#bn3/moving_mean/Read/ReadVariableOpReadVariableOpbn3/moving_mean*
_output_shapes
:*
dtype0
h
bn3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
bn3/beta
a
bn3/beta/Read/ReadVariableOpReadVariableOpbn3/beta*
_output_shapes
:*
dtype0
j
	bn3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	bn3/gamma
c
bn3/gamma/Read/ReadVariableOpReadVariableOp	bn3/gamma*
_output_shapes
:*
dtype0
l

conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
conv3/bias
e
conv3/bias/Read/ReadVariableOpReadVariableOp
conv3/bias*
_output_shapes
:*
dtype0
|
conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3/kernel
u
 conv3/kernel/Read/ReadVariableOpReadVariableOpconv3/kernel*&
_output_shapes
:*
dtype0
~
bn2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namebn2/moving_variance
w
'bn2/moving_variance/Read/ReadVariableOpReadVariableOpbn2/moving_variance*
_output_shapes
:*
dtype0
v
bn2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namebn2/moving_mean
o
#bn2/moving_mean/Read/ReadVariableOpReadVariableOpbn2/moving_mean*
_output_shapes
:*
dtype0
h
bn2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
bn2/beta
a
bn2/beta/Read/ReadVariableOpReadVariableOpbn2/beta*
_output_shapes
:*
dtype0
j
	bn2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	bn2/gamma
c
bn2/gamma/Read/ReadVariableOpReadVariableOp	bn2/gamma*
_output_shapes
:*
dtype0
l

conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
conv2/bias
e
conv2/bias/Read/ReadVariableOpReadVariableOp
conv2/bias*
_output_shapes
:*
dtype0
|
conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2/kernel
u
 conv2/kernel/Read/ReadVariableOpReadVariableOpconv2/kernel*&
_output_shapes
:*
dtype0
~
bn1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namebn1/moving_variance
w
'bn1/moving_variance/Read/ReadVariableOpReadVariableOpbn1/moving_variance*
_output_shapes
:*
dtype0
v
bn1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namebn1/moving_mean
o
#bn1/moving_mean/Read/ReadVariableOpReadVariableOpbn1/moving_mean*
_output_shapes
:*
dtype0
h
bn1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
bn1/beta
a
bn1/beta/Read/ReadVariableOpReadVariableOpbn1/beta*
_output_shapes
:*
dtype0
j
	bn1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	bn1/gamma
c
bn1/gamma/Read/ReadVariableOpReadVariableOp	bn1/gamma*
_output_shapes
:*
dtype0
l

conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
conv1/bias
e
conv1/bias/Read/ReadVariableOpReadVariableOp
conv1/bias*
_output_shapes
:*
dtype0
|
conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1/kernel
u
 conv1/kernel/Read/ReadVariableOpReadVariableOpconv1/kernel*&
_output_shapes
:*
dtype0
�
quant_softmax/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namequant_softmax/optimizer_step
�
0quant_softmax/optimizer_step/Read/ReadVariableOpReadVariableOpquant_softmax/optimizer_step*
_output_shapes
: *
dtype0
�
 quant_dense1/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" quant_dense1/post_activation_max
�
4quant_dense1/post_activation_max/Read/ReadVariableOpReadVariableOp quant_dense1/post_activation_max*
_output_shapes
: *
dtype0
�
 quant_dense1/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" quant_dense1/post_activation_min
�
4quant_dense1/post_activation_min/Read/ReadVariableOpReadVariableOp quant_dense1/post_activation_min*
_output_shapes
: *
dtype0
�
quant_dense1/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namequant_dense1/kernel_max
{
+quant_dense1/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense1/kernel_max*
_output_shapes
: *
dtype0
�
quant_dense1/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namequant_dense1/kernel_min
{
+quant_dense1/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense1/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense1/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namequant_dense1/optimizer_step
�
/quant_dense1/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense1/optimizer_step*
_output_shapes
: *
dtype0
�
quant_flatten/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namequant_flatten/optimizer_step
�
0quant_flatten/optimizer_step/Read/ReadVariableOpReadVariableOpquant_flatten/optimizer_step*
_output_shapes
: *
dtype0
�
quant_maxpool3/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_maxpool3/optimizer_step
�
1quant_maxpool3/optimizer_step/Read/ReadVariableOpReadVariableOpquant_maxpool3/optimizer_step*
_output_shapes
: *
dtype0
�
quant_relu3/output_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namequant_relu3/output_max
y
*quant_relu3/output_max/Read/ReadVariableOpReadVariableOpquant_relu3/output_max*
_output_shapes
: *
dtype0
�
quant_relu3/output_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namequant_relu3/output_min
y
*quant_relu3/output_min/Read/ReadVariableOpReadVariableOpquant_relu3/output_min*
_output_shapes
: *
dtype0
�
quant_relu3/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namequant_relu3/optimizer_step
�
.quant_relu3/optimizer_step/Read/ReadVariableOpReadVariableOpquant_relu3/optimizer_step*
_output_shapes
: *
dtype0
�
quant_bn3/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_bn3/optimizer_step
}
,quant_bn3/optimizer_step/Read/ReadVariableOpReadVariableOpquant_bn3/optimizer_step*
_output_shapes
: *
dtype0
�
quant_conv3/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namequant_conv3/kernel_max
}
*quant_conv3/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv3/kernel_max*
_output_shapes
:*
dtype0
�
quant_conv3/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namequant_conv3/kernel_min
}
*quant_conv3/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv3/kernel_min*
_output_shapes
:*
dtype0
�
quant_conv3/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namequant_conv3/optimizer_step
�
.quant_conv3/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv3/optimizer_step*
_output_shapes
: *
dtype0
�
quant_maxpool2/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_maxpool2/optimizer_step
�
1quant_maxpool2/optimizer_step/Read/ReadVariableOpReadVariableOpquant_maxpool2/optimizer_step*
_output_shapes
: *
dtype0
�
quant_relu2/output_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namequant_relu2/output_max
y
*quant_relu2/output_max/Read/ReadVariableOpReadVariableOpquant_relu2/output_max*
_output_shapes
: *
dtype0
�
quant_relu2/output_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namequant_relu2/output_min
y
*quant_relu2/output_min/Read/ReadVariableOpReadVariableOpquant_relu2/output_min*
_output_shapes
: *
dtype0
�
quant_relu2/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namequant_relu2/optimizer_step
�
.quant_relu2/optimizer_step/Read/ReadVariableOpReadVariableOpquant_relu2/optimizer_step*
_output_shapes
: *
dtype0
�
quant_bn2/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_bn2/optimizer_step
}
,quant_bn2/optimizer_step/Read/ReadVariableOpReadVariableOpquant_bn2/optimizer_step*
_output_shapes
: *
dtype0
�
quant_conv2/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namequant_conv2/kernel_max
}
*quant_conv2/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv2/kernel_max*
_output_shapes
:*
dtype0
�
quant_conv2/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namequant_conv2/kernel_min
}
*quant_conv2/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv2/kernel_min*
_output_shapes
:*
dtype0
�
quant_conv2/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namequant_conv2/optimizer_step
�
.quant_conv2/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv2/optimizer_step*
_output_shapes
: *
dtype0
�
quant_maxpool1/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_maxpool1/optimizer_step
�
1quant_maxpool1/optimizer_step/Read/ReadVariableOpReadVariableOpquant_maxpool1/optimizer_step*
_output_shapes
: *
dtype0
�
quant_relu1/output_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namequant_relu1/output_max
y
*quant_relu1/output_max/Read/ReadVariableOpReadVariableOpquant_relu1/output_max*
_output_shapes
: *
dtype0
�
quant_relu1/output_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namequant_relu1/output_min
y
*quant_relu1/output_min/Read/ReadVariableOpReadVariableOpquant_relu1/output_min*
_output_shapes
: *
dtype0
�
quant_relu1/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namequant_relu1/optimizer_step
�
.quant_relu1/optimizer_step/Read/ReadVariableOpReadVariableOpquant_relu1/optimizer_step*
_output_shapes
: *
dtype0
�
quant_bn1/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_bn1/optimizer_step
}
,quant_bn1/optimizer_step/Read/ReadVariableOpReadVariableOpquant_bn1/optimizer_step*
_output_shapes
: *
dtype0
�
quant_conv1/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namequant_conv1/kernel_max
}
*quant_conv1/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv1/kernel_max*
_output_shapes
:*
dtype0
�
quant_conv1/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namequant_conv1/kernel_min
}
*quant_conv1/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv1/kernel_min*
_output_shapes
:*
dtype0
�
quant_conv1/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namequant_conv1/optimizer_step
�
.quant_conv1/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv1/optimizer_step*
_output_shapes
: *
dtype0
�
quantize_layer/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequantize_layer/optimizer_step
�
1quantize_layer/optimizer_step/Read/ReadVariableOpReadVariableOpquantize_layer/optimizer_step*
_output_shapes
: *
dtype0
�
!quantize_layer/quantize_layer_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quantize_layer/quantize_layer_max
�
5quantize_layer/quantize_layer_max/Read/ReadVariableOpReadVariableOp!quantize_layer/quantize_layer_max*
_output_shapes
: *
dtype0
�
!quantize_layer/quantize_layer_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quantize_layer/quantize_layer_min
�
5quantize_layer/quantize_layer_min/Read/ReadVariableOpReadVariableOp!quantize_layer/quantize_layer_min*
_output_shapes
: *
dtype0
�
serving_default_inputPlaceholder*/
_output_shapes
:���������  *
dtype0*$
shape:���������  
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input!quantize_layer/quantize_layer_min!quantize_layer/quantize_layer_maxconv1/kernelquant_conv1/kernel_minquant_conv1/kernel_max
conv1/bias	bn1/gammabn1/betabn1/moving_meanbn1/moving_variancequant_relu1/output_minquant_relu1/output_maxconv2/kernelquant_conv2/kernel_minquant_conv2/kernel_max
conv2/bias	bn2/gammabn2/betabn2/moving_meanbn2/moving_variancequant_relu2/output_minquant_relu2/output_maxconv3/kernelquant_conv3/kernel_minquant_conv3/kernel_max
conv3/bias	bn3/gammabn3/betabn3/moving_meanbn3/moving_variancequant_relu3/output_minquant_relu3/output_maxdense1/kernelquant_dense1/kernel_minquant_dense1/kernel_maxdense1/bias quant_dense1/post_activation_min quant_dense1/post_activation_max*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_13349

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer_with_weights-10
layer-11
layer_with_weights-11
layer-12
layer_with_weights-12
layer-13
layer_with_weights-13
layer-14
layer_with_weights-14
layer-15
layer_with_weights-15
layer-16
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses
!quantize_layer_min
"quantize_layer_max
#quantizer_vars
$optimizer_step*
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
	+layer
,optimizer_step
-_weight_vars
.
kernel_min
/
kernel_max
0_quantize_activations
1_output_quantizers*
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
	8layer
9optimizer_step
:_weight_vars
;_quantize_activations
<_output_quantizers*
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
	Clayer
Doptimizer_step
E_weight_vars
F_quantize_activations
G_output_quantizers
H
output_min
I
output_max
J_output_quantizer_vars*
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses
	Qlayer
Roptimizer_step
S_weight_vars
T_quantize_activations
U_output_quantizers*
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses
	\layer
]optimizer_step
^_weight_vars
_
kernel_min
`
kernel_max
a_quantize_activations
b_output_quantizers*
�
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses
	ilayer
joptimizer_step
k_weight_vars
l_quantize_activations
m_output_quantizers*
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses
	tlayer
uoptimizer_step
v_weight_vars
w_quantize_activations
x_output_quantizers
y
output_min
z
output_max
{_output_quantizer_vars*
�
|	variables
}trainable_variables
~regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�layer
�optimizer_step
�_weight_vars
�_quantize_activations
�_output_quantizers*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�layer
�optimizer_step
�_weight_vars
�
kernel_min
�
kernel_max
�_quantize_activations
�_output_quantizers*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�layer
�optimizer_step
�_weight_vars
�_quantize_activations
�_output_quantizers*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�layer
�optimizer_step
�_weight_vars
�_quantize_activations
�_output_quantizers
�
output_min
�
output_max
�_output_quantizer_vars*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�layer
�optimizer_step
�_weight_vars
�_quantize_activations
�_output_quantizers*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�layer
�optimizer_step
�_weight_vars
�_quantize_activations
�_output_quantizers*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�layer
�optimizer_step
�_weight_vars
�
kernel_min
�
kernel_max
�_quantize_activations
�post_activation_min
�post_activation_max
�_output_quantizers*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�layer
�optimizer_step
�_weight_vars
�_quantize_activations
�_output_quantizers*
�
!0
"1
$2
�3
�4
,5
.6
/7
�8
�9
�10
�11
912
D13
H14
I15
R16
�17
�18
]19
_20
`21
�22
�23
�24
�25
j26
u27
y28
z29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53*
x
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*

�serving_default* 

!0
"1
$2*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
}w
VARIABLE_VALUE!quantize_layer/quantize_layer_minBlayer_with_weights-0/quantize_layer_min/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!quantize_layer/quantize_layer_maxBlayer_with_weights-0/quantize_layer_max/.ATTRIBUTES/VARIABLE_VALUE*

!min_var
"max_var*
uo
VARIABLE_VALUEquantize_layer/optimizer_step>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
)
�0
�1
,2
.3
/4*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
rl
VARIABLE_VALUEquant_conv1/optimizer_step>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

�0*
jd
VARIABLE_VALUEquant_conv1/kernel_min:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEquant_conv1/kernel_max:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
+
�0
�1
�2
�3
94*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
pj
VARIABLE_VALUEquant_bn1/optimizer_step>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

D0
H1
I2*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
rl
VARIABLE_VALUEquant_relu1/optimizer_step>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
jd
VARIABLE_VALUEquant_relu1/output_min:layer_with_weights-3/output_min/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEquant_relu1/output_max:layer_with_weights-3/output_max/.ATTRIBUTES/VARIABLE_VALUE*

Hmin_var
Imax_var*

R0*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
uo
VARIABLE_VALUEquant_maxpool1/optimizer_step>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
)
�0
�1
]2
_3
`4*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
rl
VARIABLE_VALUEquant_conv2/optimizer_step>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

�0*
jd
VARIABLE_VALUEquant_conv2/kernel_min:layer_with_weights-5/kernel_min/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEquant_conv2/kernel_max:layer_with_weights-5/kernel_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
+
�0
�1
�2
�3
j4*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
pj
VARIABLE_VALUEquant_bn2/optimizer_step>layer_with_weights-6/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

u0
y1
z2*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
rl
VARIABLE_VALUEquant_relu2/optimizer_step>layer_with_weights-7/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
jd
VARIABLE_VALUEquant_relu2/output_min:layer_with_weights-7/output_min/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEquant_relu2/output_max:layer_with_weights-7/output_max/.ATTRIBUTES/VARIABLE_VALUE*

ymin_var
zmax_var*

�0*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
|	variables
}trainable_variables
~regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
uo
VARIABLE_VALUEquant_maxpool2/optimizer_step>layer_with_weights-8/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
,
�0
�1
�2
�3
�4*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
rl
VARIABLE_VALUEquant_conv3/optimizer_step>layer_with_weights-9/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

�0*
jd
VARIABLE_VALUEquant_conv3/kernel_min:layer_with_weights-9/kernel_min/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEquant_conv3/kernel_max:layer_with_weights-9/kernel_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
,
�0
�1
�2
�3
�4*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
qk
VARIABLE_VALUEquant_bn3/optimizer_step?layer_with_weights-10/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

�0
�1
�2*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
sm
VARIABLE_VALUEquant_relu3/optimizer_step?layer_with_weights-11/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
ke
VARIABLE_VALUEquant_relu3/output_min;layer_with_weights-11/output_min/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEquant_relu3/output_max;layer_with_weights-11/output_max/.ATTRIBUTES/VARIABLE_VALUE*
 
�min_var
�max_var*

�0*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
vp
VARIABLE_VALUEquant_maxpool3/optimizer_step?layer_with_weights-12/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

�0*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
uo
VARIABLE_VALUEquant_flatten/optimizer_step?layer_with_weights-13/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
<
�0
�1
�2
�3
�4
�5
�6*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
tn
VARIABLE_VALUEquant_dense1/optimizer_step?layer_with_weights-14/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

�0*
lf
VARIABLE_VALUEquant_dense1/kernel_min;layer_with_weights-14/kernel_min/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEquant_dense1/kernel_max;layer_with_weights-14/kernel_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
~x
VARIABLE_VALUE quant_dense1/post_activation_minDlayer_with_weights-14/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE quant_dense1/post_activation_maxDlayer_with_weights-14/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
uo
VARIABLE_VALUEquant_softmax/optimizer_step?layer_with_weights-15/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
LF
VARIABLE_VALUEconv1/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
conv1/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUE	bn1/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
HB
VARIABLE_VALUEbn1/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEbn1/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbn1/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2/kernel'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
conv2/bias'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE	bn2/gamma'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEbn2/beta'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEbn2/moving_mean'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbn2/moving_variance'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv3/kernel'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
conv3/bias'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE	bn3/gamma'variables/36/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEbn3/beta'variables/37/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEbn3/moving_mean'variables/38/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbn3/moving_variance'variables/39/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense1/kernel'variables/46/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense1/bias'variables/47/.ATTRIBUTES/VARIABLE_VALUE*
�
!0
"1
$2
,3
.4
/5
�6
�7
98
D9
H10
I11
R12
]13
_14
`15
�16
�17
j18
u19
y20
z21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39*
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16*

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
x
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13*
x
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13*
* 
* 

!0
"1
$2*
* 
* 
* 
* 
* 
* 
* 
* 

,0
.1
/2*

+0*
* 
* 
* 
* 
* 
* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

�0
�2*

�0
�1
92*

80*
* 
* 
* 
* 
* 
* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

D0
H1
I2*
	
C0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 

R0*
	
Q0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

]0
_1
`2*

\0*
* 
* 
* 
* 
* 
* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

�0
�2*

�0
�1
j2*

i0*
* 
* 
* 
* 
* 
* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

u0
y1
z2*
	
t0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 

�0*


�0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1
�2*

�0*
* 
* 
* 
* 
* 
* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

�0
�2*

�0
�1
�2*

�0*
* 
* 
* 
* 
* 
* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0
�1
�2*


�0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 

�0*


�0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0*


�0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
,
�0
�1
�2
�3
�4*

�0*
* 
* 
* 
* 
* 
* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

�0
�2*

�0*


�0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
^X
VARIABLE_VALUEAdam/m/conv1/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/conv1/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/conv1/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/conv1/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/m/bn1/gamma1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/bn1/gamma1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/m/bn1/beta1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/v/bn1/beta1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/conv2/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv2/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/conv2/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/conv2/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/bn2/gamma2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/bn2/gamma2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/m/bn2/beta2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/bn2/beta2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/conv3/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv3/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/conv3/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/conv3/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/bn3/gamma2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/bn3/gamma2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/m/bn3/beta2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/bn3/beta2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense1/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense1/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense1/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense1/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 

.min_var
/max_var*

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

_min_var
`max_var*

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
 
�min_var
�max_var*

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
 
�min_var
�max_var*
* 
* 
* 
* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!quantize_layer/quantize_layer_min!quantize_layer/quantize_layer_maxquantize_layer/optimizer_stepquant_conv1/optimizer_stepquant_conv1/kernel_minquant_conv1/kernel_maxquant_bn1/optimizer_stepquant_relu1/optimizer_stepquant_relu1/output_minquant_relu1/output_maxquant_maxpool1/optimizer_stepquant_conv2/optimizer_stepquant_conv2/kernel_minquant_conv2/kernel_maxquant_bn2/optimizer_stepquant_relu2/optimizer_stepquant_relu2/output_minquant_relu2/output_maxquant_maxpool2/optimizer_stepquant_conv3/optimizer_stepquant_conv3/kernel_minquant_conv3/kernel_maxquant_bn3/optimizer_stepquant_relu3/optimizer_stepquant_relu3/output_minquant_relu3/output_maxquant_maxpool3/optimizer_stepquant_flatten/optimizer_stepquant_dense1/optimizer_stepquant_dense1/kernel_minquant_dense1/kernel_max quant_dense1/post_activation_min quant_dense1/post_activation_maxquant_softmax/optimizer_stepconv1/kernel
conv1/bias	bn1/gammabn1/betabn1/moving_meanbn1/moving_varianceconv2/kernel
conv2/bias	bn2/gammabn2/betabn2/moving_meanbn2/moving_varianceconv3/kernel
conv3/bias	bn3/gammabn3/betabn3/moving_meanbn3/moving_variancedense1/kerneldense1/bias	iterationlearning_rateAdam/m/conv1/kernelAdam/v/conv1/kernelAdam/m/conv1/biasAdam/v/conv1/biasAdam/m/bn1/gammaAdam/v/bn1/gammaAdam/m/bn1/betaAdam/v/bn1/betaAdam/m/conv2/kernelAdam/v/conv2/kernelAdam/m/conv2/biasAdam/v/conv2/biasAdam/m/bn2/gammaAdam/v/bn2/gammaAdam/m/bn2/betaAdam/v/bn2/betaAdam/m/conv3/kernelAdam/v/conv3/kernelAdam/m/conv3/biasAdam/v/conv3/biasAdam/m/bn3/gammaAdam/v/bn3/gammaAdam/m/bn3/betaAdam/v/bn3/betaAdam/m/dense1/kernelAdam/v/dense1/kernelAdam/m/dense1/biasAdam/v/dense1/biastotal_1count_1totalcountConst*e
Tin^
\2Z*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_15454
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename!quantize_layer/quantize_layer_min!quantize_layer/quantize_layer_maxquantize_layer/optimizer_stepquant_conv1/optimizer_stepquant_conv1/kernel_minquant_conv1/kernel_maxquant_bn1/optimizer_stepquant_relu1/optimizer_stepquant_relu1/output_minquant_relu1/output_maxquant_maxpool1/optimizer_stepquant_conv2/optimizer_stepquant_conv2/kernel_minquant_conv2/kernel_maxquant_bn2/optimizer_stepquant_relu2/optimizer_stepquant_relu2/output_minquant_relu2/output_maxquant_maxpool2/optimizer_stepquant_conv3/optimizer_stepquant_conv3/kernel_minquant_conv3/kernel_maxquant_bn3/optimizer_stepquant_relu3/optimizer_stepquant_relu3/output_minquant_relu3/output_maxquant_maxpool3/optimizer_stepquant_flatten/optimizer_stepquant_dense1/optimizer_stepquant_dense1/kernel_minquant_dense1/kernel_max quant_dense1/post_activation_min quant_dense1/post_activation_maxquant_softmax/optimizer_stepconv1/kernel
conv1/bias	bn1/gammabn1/betabn1/moving_meanbn1/moving_varianceconv2/kernel
conv2/bias	bn2/gammabn2/betabn2/moving_meanbn2/moving_varianceconv3/kernel
conv3/bias	bn3/gammabn3/betabn3/moving_meanbn3/moving_variancedense1/kerneldense1/bias	iterationlearning_rateAdam/m/conv1/kernelAdam/v/conv1/kernelAdam/m/conv1/biasAdam/v/conv1/biasAdam/m/bn1/gammaAdam/v/bn1/gammaAdam/m/bn1/betaAdam/v/bn1/betaAdam/m/conv2/kernelAdam/v/conv2/kernelAdam/m/conv2/biasAdam/v/conv2/biasAdam/m/bn2/gammaAdam/v/bn2/gammaAdam/m/bn2/betaAdam/v/bn2/betaAdam/m/conv3/kernelAdam/v/conv3/kernelAdam/m/conv3/biasAdam/v/conv3/biasAdam/m/bn3/gammaAdam/v/bn3/gammaAdam/m/bn3/betaAdam/v/bn3/betaAdam/m/dense1/kernelAdam/v/dense1/kernelAdam/m/dense1/biasAdam/v/dense1/biastotal_1count_1totalcount*d
Tin]
[2Y*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_15728��!
�
�
)__inference_quant_bn1_layer_call_fn_14012

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_quant_bn1_layer_call_and_return_conditional_losses_11804w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������  : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
e
I__inference_quant_maxpool3_layer_call_and_return_conditional_losses_12370

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
e
I__inference_quant_maxpool2_layer_call_and_return_conditional_losses_12298

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
��
�N
__inference__traced_save_15454
file_prefixB
8read_disablecopyonread_quantize_layer_quantize_layer_min: D
:read_1_disablecopyonread_quantize_layer_quantize_layer_max: @
6read_2_disablecopyonread_quantize_layer_optimizer_step: =
3read_3_disablecopyonread_quant_conv1_optimizer_step: =
/read_4_disablecopyonread_quant_conv1_kernel_min:=
/read_5_disablecopyonread_quant_conv1_kernel_max:;
1read_6_disablecopyonread_quant_bn1_optimizer_step: =
3read_7_disablecopyonread_quant_relu1_optimizer_step: 9
/read_8_disablecopyonread_quant_relu1_output_min: 9
/read_9_disablecopyonread_quant_relu1_output_max: A
7read_10_disablecopyonread_quant_maxpool1_optimizer_step: >
4read_11_disablecopyonread_quant_conv2_optimizer_step: >
0read_12_disablecopyonread_quant_conv2_kernel_min:>
0read_13_disablecopyonread_quant_conv2_kernel_max:<
2read_14_disablecopyonread_quant_bn2_optimizer_step: >
4read_15_disablecopyonread_quant_relu2_optimizer_step: :
0read_16_disablecopyonread_quant_relu2_output_min: :
0read_17_disablecopyonread_quant_relu2_output_max: A
7read_18_disablecopyonread_quant_maxpool2_optimizer_step: >
4read_19_disablecopyonread_quant_conv3_optimizer_step: >
0read_20_disablecopyonread_quant_conv3_kernel_min:>
0read_21_disablecopyonread_quant_conv3_kernel_max:<
2read_22_disablecopyonread_quant_bn3_optimizer_step: >
4read_23_disablecopyonread_quant_relu3_optimizer_step: :
0read_24_disablecopyonread_quant_relu3_output_min: :
0read_25_disablecopyonread_quant_relu3_output_max: A
7read_26_disablecopyonread_quant_maxpool3_optimizer_step: @
6read_27_disablecopyonread_quant_flatten_optimizer_step: ?
5read_28_disablecopyonread_quant_dense1_optimizer_step: ;
1read_29_disablecopyonread_quant_dense1_kernel_min: ;
1read_30_disablecopyonread_quant_dense1_kernel_max: D
:read_31_disablecopyonread_quant_dense1_post_activation_min: D
:read_32_disablecopyonread_quant_dense1_post_activation_max: @
6read_33_disablecopyonread_quant_softmax_optimizer_step: @
&read_34_disablecopyonread_conv1_kernel:2
$read_35_disablecopyonread_conv1_bias:1
#read_36_disablecopyonread_bn1_gamma:0
"read_37_disablecopyonread_bn1_beta:7
)read_38_disablecopyonread_bn1_moving_mean:;
-read_39_disablecopyonread_bn1_moving_variance:@
&read_40_disablecopyonread_conv2_kernel:2
$read_41_disablecopyonread_conv2_bias:1
#read_42_disablecopyonread_bn2_gamma:0
"read_43_disablecopyonread_bn2_beta:7
)read_44_disablecopyonread_bn2_moving_mean:;
-read_45_disablecopyonread_bn2_moving_variance:@
&read_46_disablecopyonread_conv3_kernel:2
$read_47_disablecopyonread_conv3_bias:1
#read_48_disablecopyonread_bn3_gamma:0
"read_49_disablecopyonread_bn3_beta:7
)read_50_disablecopyonread_bn3_moving_mean:;
-read_51_disablecopyonread_bn3_moving_variance::
'read_52_disablecopyonread_dense1_kernel:	�
3
%read_53_disablecopyonread_dense1_bias:
-
#read_54_disablecopyonread_iteration:	 1
'read_55_disablecopyonread_learning_rate: G
-read_56_disablecopyonread_adam_m_conv1_kernel:G
-read_57_disablecopyonread_adam_v_conv1_kernel:9
+read_58_disablecopyonread_adam_m_conv1_bias:9
+read_59_disablecopyonread_adam_v_conv1_bias:8
*read_60_disablecopyonread_adam_m_bn1_gamma:8
*read_61_disablecopyonread_adam_v_bn1_gamma:7
)read_62_disablecopyonread_adam_m_bn1_beta:7
)read_63_disablecopyonread_adam_v_bn1_beta:G
-read_64_disablecopyonread_adam_m_conv2_kernel:G
-read_65_disablecopyonread_adam_v_conv2_kernel:9
+read_66_disablecopyonread_adam_m_conv2_bias:9
+read_67_disablecopyonread_adam_v_conv2_bias:8
*read_68_disablecopyonread_adam_m_bn2_gamma:8
*read_69_disablecopyonread_adam_v_bn2_gamma:7
)read_70_disablecopyonread_adam_m_bn2_beta:7
)read_71_disablecopyonread_adam_v_bn2_beta:G
-read_72_disablecopyonread_adam_m_conv3_kernel:G
-read_73_disablecopyonread_adam_v_conv3_kernel:9
+read_74_disablecopyonread_adam_m_conv3_bias:9
+read_75_disablecopyonread_adam_v_conv3_bias:8
*read_76_disablecopyonread_adam_m_bn3_gamma:8
*read_77_disablecopyonread_adam_v_bn3_gamma:7
)read_78_disablecopyonread_adam_m_bn3_beta:7
)read_79_disablecopyonread_adam_v_bn3_beta:A
.read_80_disablecopyonread_adam_m_dense1_kernel:	�
A
.read_81_disablecopyonread_adam_v_dense1_kernel:	�
:
,read_82_disablecopyonread_adam_m_dense1_bias:
:
,read_83_disablecopyonread_adam_v_dense1_bias:
+
!read_84_disablecopyonread_total_1: +
!read_85_disablecopyonread_count_1: )
read_86_disablecopyonread_total: )
read_87_disablecopyonread_count: 
savev2_const
identity_177��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_66/DisableCopyOnRead�Read_66/ReadVariableOp�Read_67/DisableCopyOnRead�Read_67/ReadVariableOp�Read_68/DisableCopyOnRead�Read_68/ReadVariableOp�Read_69/DisableCopyOnRead�Read_69/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_70/DisableCopyOnRead�Read_70/ReadVariableOp�Read_71/DisableCopyOnRead�Read_71/ReadVariableOp�Read_72/DisableCopyOnRead�Read_72/ReadVariableOp�Read_73/DisableCopyOnRead�Read_73/ReadVariableOp�Read_74/DisableCopyOnRead�Read_74/ReadVariableOp�Read_75/DisableCopyOnRead�Read_75/ReadVariableOp�Read_76/DisableCopyOnRead�Read_76/ReadVariableOp�Read_77/DisableCopyOnRead�Read_77/ReadVariableOp�Read_78/DisableCopyOnRead�Read_78/ReadVariableOp�Read_79/DisableCopyOnRead�Read_79/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_80/DisableCopyOnRead�Read_80/ReadVariableOp�Read_81/DisableCopyOnRead�Read_81/ReadVariableOp�Read_82/DisableCopyOnRead�Read_82/ReadVariableOp�Read_83/DisableCopyOnRead�Read_83/ReadVariableOp�Read_84/DisableCopyOnRead�Read_84/ReadVariableOp�Read_85/DisableCopyOnRead�Read_85/ReadVariableOp�Read_86/DisableCopyOnRead�Read_86/ReadVariableOp�Read_87/DisableCopyOnRead�Read_87/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
Read/DisableCopyOnReadDisableCopyOnRead8read_disablecopyonread_quantize_layer_quantize_layer_min"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp8read_disablecopyonread_quantize_layer_quantize_layer_min^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0a
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: Y

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_1/DisableCopyOnReadDisableCopyOnRead:read_1_disablecopyonread_quantize_layer_quantize_layer_max"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp:read_1_disablecopyonread_quantize_layer_quantize_layer_max^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: [

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_2/DisableCopyOnReadDisableCopyOnRead6read_2_disablecopyonread_quantize_layer_optimizer_step"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp6read_2_disablecopyonread_quantize_layer_optimizer_step^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: [

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_3/DisableCopyOnReadDisableCopyOnRead3read_3_disablecopyonread_quant_conv1_optimizer_step"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp3read_3_disablecopyonread_quant_conv1_optimizer_step^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: [

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_4/DisableCopyOnReadDisableCopyOnRead/read_4_disablecopyonread_quant_conv1_kernel_min"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp/read_4_disablecopyonread_quant_conv1_kernel_min^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_5/DisableCopyOnReadDisableCopyOnRead/read_5_disablecopyonread_quant_conv1_kernel_max"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp/read_5_disablecopyonread_quant_conv1_kernel_max^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_6/DisableCopyOnReadDisableCopyOnRead1read_6_disablecopyonread_quant_bn1_optimizer_step"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp1read_6_disablecopyonread_quant_bn1_optimizer_step^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_7/DisableCopyOnReadDisableCopyOnRead3read_7_disablecopyonread_quant_relu1_optimizer_step"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp3read_7_disablecopyonread_quant_relu1_optimizer_step^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_8/DisableCopyOnReadDisableCopyOnRead/read_8_disablecopyonread_quant_relu1_output_min"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp/read_8_disablecopyonread_quant_relu1_output_min^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_9/DisableCopyOnReadDisableCopyOnRead/read_9_disablecopyonread_quant_relu1_output_max"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp/read_9_disablecopyonread_quant_relu1_output_max^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_10/DisableCopyOnReadDisableCopyOnRead7read_10_disablecopyonread_quant_maxpool1_optimizer_step"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp7read_10_disablecopyonread_quant_maxpool1_optimizer_step^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_11/DisableCopyOnReadDisableCopyOnRead4read_11_disablecopyonread_quant_conv2_optimizer_step"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp4read_11_disablecopyonread_quant_conv2_optimizer_step^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_12/DisableCopyOnReadDisableCopyOnRead0read_12_disablecopyonread_quant_conv2_kernel_min"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp0read_12_disablecopyonread_quant_conv2_kernel_min^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_13/DisableCopyOnReadDisableCopyOnRead0read_13_disablecopyonread_quant_conv2_kernel_max"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp0read_13_disablecopyonread_quant_conv2_kernel_max^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_14/DisableCopyOnReadDisableCopyOnRead2read_14_disablecopyonread_quant_bn2_optimizer_step"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp2read_14_disablecopyonread_quant_bn2_optimizer_step^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_15/DisableCopyOnReadDisableCopyOnRead4read_15_disablecopyonread_quant_relu2_optimizer_step"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp4read_15_disablecopyonread_quant_relu2_optimizer_step^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_16/DisableCopyOnReadDisableCopyOnRead0read_16_disablecopyonread_quant_relu2_output_min"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp0read_16_disablecopyonread_quant_relu2_output_min^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_17/DisableCopyOnReadDisableCopyOnRead0read_17_disablecopyonread_quant_relu2_output_max"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp0read_17_disablecopyonread_quant_relu2_output_max^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_18/DisableCopyOnReadDisableCopyOnRead7read_18_disablecopyonread_quant_maxpool2_optimizer_step"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp7read_18_disablecopyonread_quant_maxpool2_optimizer_step^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_19/DisableCopyOnReadDisableCopyOnRead4read_19_disablecopyonread_quant_conv3_optimizer_step"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp4read_19_disablecopyonread_quant_conv3_optimizer_step^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_20/DisableCopyOnReadDisableCopyOnRead0read_20_disablecopyonread_quant_conv3_kernel_min"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp0read_20_disablecopyonread_quant_conv3_kernel_min^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_21/DisableCopyOnReadDisableCopyOnRead0read_21_disablecopyonread_quant_conv3_kernel_max"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp0read_21_disablecopyonread_quant_conv3_kernel_max^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_22/DisableCopyOnReadDisableCopyOnRead2read_22_disablecopyonread_quant_bn3_optimizer_step"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp2read_22_disablecopyonread_quant_bn3_optimizer_step^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_23/DisableCopyOnReadDisableCopyOnRead4read_23_disablecopyonread_quant_relu3_optimizer_step"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp4read_23_disablecopyonread_quant_relu3_optimizer_step^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_24/DisableCopyOnReadDisableCopyOnRead0read_24_disablecopyonread_quant_relu3_output_min"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp0read_24_disablecopyonread_quant_relu3_output_min^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_25/DisableCopyOnReadDisableCopyOnRead0read_25_disablecopyonread_quant_relu3_output_max"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp0read_25_disablecopyonread_quant_relu3_output_max^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_26/DisableCopyOnReadDisableCopyOnRead7read_26_disablecopyonread_quant_maxpool3_optimizer_step"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp7read_26_disablecopyonread_quant_maxpool3_optimizer_step^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_27/DisableCopyOnReadDisableCopyOnRead6read_27_disablecopyonread_quant_flatten_optimizer_step"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp6read_27_disablecopyonread_quant_flatten_optimizer_step^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_28/DisableCopyOnReadDisableCopyOnRead5read_28_disablecopyonread_quant_dense1_optimizer_step"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp5read_28_disablecopyonread_quant_dense1_optimizer_step^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_29/DisableCopyOnReadDisableCopyOnRead1read_29_disablecopyonread_quant_dense1_kernel_min"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp1read_29_disablecopyonread_quant_dense1_kernel_min^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_30/DisableCopyOnReadDisableCopyOnRead1read_30_disablecopyonread_quant_dense1_kernel_max"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp1read_30_disablecopyonread_quant_dense1_kernel_max^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_31/DisableCopyOnReadDisableCopyOnRead:read_31_disablecopyonread_quant_dense1_post_activation_min"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp:read_31_disablecopyonread_quant_dense1_post_activation_min^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_32/DisableCopyOnReadDisableCopyOnRead:read_32_disablecopyonread_quant_dense1_post_activation_max"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp:read_32_disablecopyonread_quant_dense1_post_activation_max^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_33/DisableCopyOnReadDisableCopyOnRead6read_33_disablecopyonread_quant_softmax_optimizer_step"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp6read_33_disablecopyonread_quant_softmax_optimizer_step^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
: {
Read_34/DisableCopyOnReadDisableCopyOnRead&read_34_disablecopyonread_conv1_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp&read_34_disablecopyonread_conv1_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*&
_output_shapes
:y
Read_35/DisableCopyOnReadDisableCopyOnRead$read_35_disablecopyonread_conv1_bias"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp$read_35_disablecopyonread_conv1_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_36/DisableCopyOnReadDisableCopyOnRead#read_36_disablecopyonread_bn1_gamma"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp#read_36_disablecopyonread_bn1_gamma^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:w
Read_37/DisableCopyOnReadDisableCopyOnRead"read_37_disablecopyonread_bn1_beta"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp"read_37_disablecopyonread_bn1_beta^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_38/DisableCopyOnReadDisableCopyOnRead)read_38_disablecopyonread_bn1_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp)read_38_disablecopyonread_bn1_moving_mean^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_39/DisableCopyOnReadDisableCopyOnRead-read_39_disablecopyonread_bn1_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp-read_39_disablecopyonread_bn1_moving_variance^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:{
Read_40/DisableCopyOnReadDisableCopyOnRead&read_40_disablecopyonread_conv2_kernel"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp&read_40_disablecopyonread_conv2_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*&
_output_shapes
:y
Read_41/DisableCopyOnReadDisableCopyOnRead$read_41_disablecopyonread_conv2_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp$read_41_disablecopyonread_conv2_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_42/DisableCopyOnReadDisableCopyOnRead#read_42_disablecopyonread_bn2_gamma"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp#read_42_disablecopyonread_bn2_gamma^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:w
Read_43/DisableCopyOnReadDisableCopyOnRead"read_43_disablecopyonread_bn2_beta"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp"read_43_disablecopyonread_bn2_beta^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_44/DisableCopyOnReadDisableCopyOnRead)read_44_disablecopyonread_bn2_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp)read_44_disablecopyonread_bn2_moving_mean^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_45/DisableCopyOnReadDisableCopyOnRead-read_45_disablecopyonread_bn2_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp-read_45_disablecopyonread_bn2_moving_variance^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:{
Read_46/DisableCopyOnReadDisableCopyOnRead&read_46_disablecopyonread_conv3_kernel"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp&read_46_disablecopyonread_conv3_kernel^Read_46/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*&
_output_shapes
:y
Read_47/DisableCopyOnReadDisableCopyOnRead$read_47_disablecopyonread_conv3_bias"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp$read_47_disablecopyonread_conv3_bias^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_48/DisableCopyOnReadDisableCopyOnRead#read_48_disablecopyonread_bn3_gamma"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp#read_48_disablecopyonread_bn3_gamma^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
:w
Read_49/DisableCopyOnReadDisableCopyOnRead"read_49_disablecopyonread_bn3_beta"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp"read_49_disablecopyonread_bn3_beta^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_50/DisableCopyOnReadDisableCopyOnRead)read_50_disablecopyonread_bn3_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp)read_50_disablecopyonread_bn3_moving_mean^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_51/DisableCopyOnReadDisableCopyOnRead-read_51_disablecopyonread_bn3_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp-read_51_disablecopyonread_bn3_moving_variance^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_52/DisableCopyOnReadDisableCopyOnRead'read_52_disablecopyonread_dense1_kernel"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp'read_52_disablecopyonread_dense1_kernel^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�
*
dtype0q
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�
h
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
z
Read_53/DisableCopyOnReadDisableCopyOnRead%read_53_disablecopyonread_dense1_bias"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp%read_53_disablecopyonread_dense1_bias^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0l
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
c
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
:
x
Read_54/DisableCopyOnReadDisableCopyOnRead#read_54_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp#read_54_disablecopyonread_iteration^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	h
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: _
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_55/DisableCopyOnReadDisableCopyOnRead'read_55_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp'read_55_disablecopyonread_learning_rate^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_56/DisableCopyOnReadDisableCopyOnRead-read_56_disablecopyonread_adam_m_conv1_kernel"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp-read_56_disablecopyonread_adam_m_conv1_kernel^Read_56/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_57/DisableCopyOnReadDisableCopyOnRead-read_57_disablecopyonread_adam_v_conv1_kernel"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp-read_57_disablecopyonread_adam_v_conv1_kernel^Read_57/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_58/DisableCopyOnReadDisableCopyOnRead+read_58_disablecopyonread_adam_m_conv1_bias"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOp+read_58_disablecopyonread_adam_m_conv1_bias^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_59/DisableCopyOnReadDisableCopyOnRead+read_59_disablecopyonread_adam_v_conv1_bias"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOp+read_59_disablecopyonread_adam_v_conv1_bias^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_60/DisableCopyOnReadDisableCopyOnRead*read_60_disablecopyonread_adam_m_bn1_gamma"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp*read_60_disablecopyonread_adam_m_bn1_gamma^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_61/DisableCopyOnReadDisableCopyOnRead*read_61_disablecopyonread_adam_v_bn1_gamma"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp*read_61_disablecopyonread_adam_v_bn1_gamma^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_62/DisableCopyOnReadDisableCopyOnRead)read_62_disablecopyonread_adam_m_bn1_beta"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOp)read_62_disablecopyonread_adam_m_bn1_beta^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_63/DisableCopyOnReadDisableCopyOnRead)read_63_disablecopyonread_adam_v_bn1_beta"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOp)read_63_disablecopyonread_adam_v_bn1_beta^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_64/DisableCopyOnReadDisableCopyOnRead-read_64_disablecopyonread_adam_m_conv2_kernel"/device:CPU:0*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOp-read_64_disablecopyonread_adam_m_conv2_kernel^Read_64/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_65/DisableCopyOnReadDisableCopyOnRead-read_65_disablecopyonread_adam_v_conv2_kernel"/device:CPU:0*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOp-read_65_disablecopyonread_adam_v_conv2_kernel^Read_65/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_66/DisableCopyOnReadDisableCopyOnRead+read_66_disablecopyonread_adam_m_conv2_bias"/device:CPU:0*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOp+read_66_disablecopyonread_adam_m_conv2_bias^Read_66/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_67/DisableCopyOnReadDisableCopyOnRead+read_67_disablecopyonread_adam_v_conv2_bias"/device:CPU:0*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOp+read_67_disablecopyonread_adam_v_conv2_bias^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_68/DisableCopyOnReadDisableCopyOnRead*read_68_disablecopyonread_adam_m_bn2_gamma"/device:CPU:0*
_output_shapes
 �
Read_68/ReadVariableOpReadVariableOp*read_68_disablecopyonread_adam_m_bn2_gamma^Read_68/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_69/DisableCopyOnReadDisableCopyOnRead*read_69_disablecopyonread_adam_v_bn2_gamma"/device:CPU:0*
_output_shapes
 �
Read_69/ReadVariableOpReadVariableOp*read_69_disablecopyonread_adam_v_bn2_gamma^Read_69/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_70/DisableCopyOnReadDisableCopyOnRead)read_70_disablecopyonread_adam_m_bn2_beta"/device:CPU:0*
_output_shapes
 �
Read_70/ReadVariableOpReadVariableOp)read_70_disablecopyonread_adam_m_bn2_beta^Read_70/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_71/DisableCopyOnReadDisableCopyOnRead)read_71_disablecopyonread_adam_v_bn2_beta"/device:CPU:0*
_output_shapes
 �
Read_71/ReadVariableOpReadVariableOp)read_71_disablecopyonread_adam_v_bn2_beta^Read_71/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_72/DisableCopyOnReadDisableCopyOnRead-read_72_disablecopyonread_adam_m_conv3_kernel"/device:CPU:0*
_output_shapes
 �
Read_72/ReadVariableOpReadVariableOp-read_72_disablecopyonread_adam_m_conv3_kernel^Read_72/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_73/DisableCopyOnReadDisableCopyOnRead-read_73_disablecopyonread_adam_v_conv3_kernel"/device:CPU:0*
_output_shapes
 �
Read_73/ReadVariableOpReadVariableOp-read_73_disablecopyonread_adam_v_conv3_kernel^Read_73/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_74/DisableCopyOnReadDisableCopyOnRead+read_74_disablecopyonread_adam_m_conv3_bias"/device:CPU:0*
_output_shapes
 �
Read_74/ReadVariableOpReadVariableOp+read_74_disablecopyonread_adam_m_conv3_bias^Read_74/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_148IdentityRead_74/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_75/DisableCopyOnReadDisableCopyOnRead+read_75_disablecopyonread_adam_v_conv3_bias"/device:CPU:0*
_output_shapes
 �
Read_75/ReadVariableOpReadVariableOp+read_75_disablecopyonread_adam_v_conv3_bias^Read_75/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_150IdentityRead_75/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_76/DisableCopyOnReadDisableCopyOnRead*read_76_disablecopyonread_adam_m_bn3_gamma"/device:CPU:0*
_output_shapes
 �
Read_76/ReadVariableOpReadVariableOp*read_76_disablecopyonread_adam_m_bn3_gamma^Read_76/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_152IdentityRead_76/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_77/DisableCopyOnReadDisableCopyOnRead*read_77_disablecopyonread_adam_v_bn3_gamma"/device:CPU:0*
_output_shapes
 �
Read_77/ReadVariableOpReadVariableOp*read_77_disablecopyonread_adam_v_bn3_gamma^Read_77/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_154IdentityRead_77/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_78/DisableCopyOnReadDisableCopyOnRead)read_78_disablecopyonread_adam_m_bn3_beta"/device:CPU:0*
_output_shapes
 �
Read_78/ReadVariableOpReadVariableOp)read_78_disablecopyonread_adam_m_bn3_beta^Read_78/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_156IdentityRead_78/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_157IdentityIdentity_156:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_79/DisableCopyOnReadDisableCopyOnRead)read_79_disablecopyonread_adam_v_bn3_beta"/device:CPU:0*
_output_shapes
 �
Read_79/ReadVariableOpReadVariableOp)read_79_disablecopyonread_adam_v_bn3_beta^Read_79/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_158IdentityRead_79/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_159IdentityIdentity_158:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_80/DisableCopyOnReadDisableCopyOnRead.read_80_disablecopyonread_adam_m_dense1_kernel"/device:CPU:0*
_output_shapes
 �
Read_80/ReadVariableOpReadVariableOp.read_80_disablecopyonread_adam_m_dense1_kernel^Read_80/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�
*
dtype0q
Identity_160IdentityRead_80/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�
h
Identity_161IdentityIdentity_160:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
�
Read_81/DisableCopyOnReadDisableCopyOnRead.read_81_disablecopyonread_adam_v_dense1_kernel"/device:CPU:0*
_output_shapes
 �
Read_81/ReadVariableOpReadVariableOp.read_81_disablecopyonread_adam_v_dense1_kernel^Read_81/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�
*
dtype0q
Identity_162IdentityRead_81/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�
h
Identity_163IdentityIdentity_162:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
�
Read_82/DisableCopyOnReadDisableCopyOnRead,read_82_disablecopyonread_adam_m_dense1_bias"/device:CPU:0*
_output_shapes
 �
Read_82/ReadVariableOpReadVariableOp,read_82_disablecopyonread_adam_m_dense1_bias^Read_82/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0l
Identity_164IdentityRead_82/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
c
Identity_165IdentityIdentity_164:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_83/DisableCopyOnReadDisableCopyOnRead,read_83_disablecopyonread_adam_v_dense1_bias"/device:CPU:0*
_output_shapes
 �
Read_83/ReadVariableOpReadVariableOp,read_83_disablecopyonread_adam_v_dense1_bias^Read_83/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0l
Identity_166IdentityRead_83/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
c
Identity_167IdentityIdentity_166:output:0"/device:CPU:0*
T0*
_output_shapes
:
v
Read_84/DisableCopyOnReadDisableCopyOnRead!read_84_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_84/ReadVariableOpReadVariableOp!read_84_disablecopyonread_total_1^Read_84/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_168IdentityRead_84/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_169IdentityIdentity_168:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_85/DisableCopyOnReadDisableCopyOnRead!read_85_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_85/ReadVariableOpReadVariableOp!read_85_disablecopyonread_count_1^Read_85/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_170IdentityRead_85/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_171IdentityIdentity_170:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_86/DisableCopyOnReadDisableCopyOnReadread_86_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_86/ReadVariableOpReadVariableOpread_86_disablecopyonread_total^Read_86/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_172IdentityRead_86/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_173IdentityIdentity_172:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_87/DisableCopyOnReadDisableCopyOnReadread_87_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_87/ReadVariableOpReadVariableOpread_87_disablecopyonread_count^Read_87/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_174IdentityRead_87/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_175IdentityIdentity_174:output:0"/device:CPU:0*
T0*
_output_shapes
: �&
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*�%
value�%B�%YBBlayer_with_weights-0/quantize_layer_min/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/quantize_layer_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/output_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/output_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-6/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-7/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/output_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/output_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-8/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-9/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-9/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-9/kernel_max/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-10/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-11/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-11/output_min/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-11/output_max/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-12/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-13/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-14/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-14/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-14/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-14/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-14/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-15/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*�
value�B�YB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0Identity_157:output:0Identity_159:output:0Identity_161:output:0Identity_163:output:0Identity_165:output:0Identity_167:output:0Identity_169:output:0Identity_171:output:0Identity_173:output:0Identity_175:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *g
dtypes]
[2Y	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_176Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_177IdentityIdentity_176:output:0^NoOp*
T0*
_output_shapes
: �$
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_78/DisableCopyOnRead^Read_78/ReadVariableOp^Read_79/DisableCopyOnRead^Read_79/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_80/DisableCopyOnRead^Read_80/ReadVariableOp^Read_81/DisableCopyOnRead^Read_81/ReadVariableOp^Read_82/DisableCopyOnRead^Read_82/ReadVariableOp^Read_83/DisableCopyOnRead^Read_83/ReadVariableOp^Read_84/DisableCopyOnRead^Read_84/ReadVariableOp^Read_85/DisableCopyOnRead^Read_85/ReadVariableOp^Read_86/DisableCopyOnRead^Read_86/ReadVariableOp^Read_87/DisableCopyOnRead^Read_87/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "%
identity_177Identity_177:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp26
Read_78/DisableCopyOnReadRead_78/DisableCopyOnRead20
Read_78/ReadVariableOpRead_78/ReadVariableOp26
Read_79/DisableCopyOnReadRead_79/DisableCopyOnRead20
Read_79/ReadVariableOpRead_79/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp26
Read_80/DisableCopyOnReadRead_80/DisableCopyOnRead20
Read_80/ReadVariableOpRead_80/ReadVariableOp26
Read_81/DisableCopyOnReadRead_81/DisableCopyOnRead20
Read_81/ReadVariableOpRead_81/ReadVariableOp26
Read_82/DisableCopyOnReadRead_82/DisableCopyOnRead20
Read_82/ReadVariableOpRead_82/ReadVariableOp26
Read_83/DisableCopyOnReadRead_83/DisableCopyOnRead20
Read_83/ReadVariableOpRead_83/ReadVariableOp26
Read_84/DisableCopyOnReadRead_84/DisableCopyOnRead20
Read_84/ReadVariableOpRead_84/ReadVariableOp26
Read_85/DisableCopyOnReadRead_85/DisableCopyOnRead20
Read_85/ReadVariableOpRead_85/ReadVariableOp26
Read_86/DisableCopyOnReadRead_86/DisableCopyOnRead20
Read_86/ReadVariableOpRead_86/ReadVariableOp26
Read_87/DisableCopyOnReadRead_87/DisableCopyOnRead20
Read_87/ReadVariableOpRead_87/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:Y

_output_shapes
: 
�
�
#__inference_bn1_layer_call_fn_14713

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_11527�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
D
(__inference_maxpool2_layer_call_fn_14826

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_maxpool2_layer_call_and_return_conditional_losses_11636�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
_
C__inference_maxpool1_layer_call_and_return_conditional_losses_11560

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
e
I__inference_quant_maxpool1_layer_call_and_return_conditional_losses_12226

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  :W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
#__inference_bn2_layer_call_fn_14772

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn2_layer_call_and_return_conditional_losses_11585�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
#__inference_bn2_layer_call_fn_14785

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn2_layer_call_and_return_conditional_losses_11603�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
+__inference_quant_relu1_layer_call_fn_14079

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_relu1_layer_call_and_return_conditional_losses_12216w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
D
(__inference_maxpool3_layer_call_fn_14898

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_maxpool3_layer_call_and_return_conditional_losses_11712�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
)__inference_MNIST_CNN_layer_call_fn_12597	
input
unknown: 
	unknown_0: #
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9: 

unknown_10: $

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19: 

unknown_20: $

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29: 

unknown_30: 

unknown_31:	�


unknown_32: 

unknown_33: 

unknown_34:


unknown_35: 

unknown_36: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*0
_read_only_resource_inputs
!$*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_MNIST_CNN_layer_call_and_return_conditional_losses_12518o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:���������  

_user_specified_nameinput
�
�
D__inference_quant_bn3_layer_call_and_return_conditional_losses_14451

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�P
�
D__inference_MNIST_CNN_layer_call_and_return_conditional_losses_12695

inputs
quantize_layer_12602: 
quantize_layer_12604: +
quant_conv1_12607:
quant_conv1_12609:
quant_conv1_12611:
quant_conv1_12613:
quant_bn1_12616:
quant_bn1_12618:
quant_bn1_12620:
quant_bn1_12622:
quant_relu1_12625: 
quant_relu1_12627: +
quant_conv2_12631:
quant_conv2_12633:
quant_conv2_12635:
quant_conv2_12637:
quant_bn2_12640:
quant_bn2_12642:
quant_bn2_12644:
quant_bn2_12646:
quant_relu2_12649: 
quant_relu2_12651: +
quant_conv3_12655:
quant_conv3_12657:
quant_conv3_12659:
quant_conv3_12661:
quant_bn3_12664:
quant_bn3_12666:
quant_bn3_12668:
quant_bn3_12670:
quant_relu3_12673: 
quant_relu3_12675: %
quant_dense1_12680:	�

quant_dense1_12682: 
quant_dense1_12684:  
quant_dense1_12686:

quant_dense1_12688: 
quant_dense1_12690: 
identity��!quant_bn1/StatefulPartitionedCall�!quant_bn2/StatefulPartitionedCall�!quant_bn3/StatefulPartitionedCall�#quant_conv1/StatefulPartitionedCall�#quant_conv2/StatefulPartitionedCall�#quant_conv3/StatefulPartitionedCall�$quant_dense1/StatefulPartitionedCall�#quant_relu1/StatefulPartitionedCall�#quant_relu2/StatefulPartitionedCall�#quant_relu3/StatefulPartitionedCall�&quantize_layer/StatefulPartitionedCall�
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_12602quantize_layer_12604*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quantize_layer_layer_call_and_return_conditional_losses_12150�
#quant_conv1/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv1_12607quant_conv1_12609quant_conv1_12611quant_conv1_12613*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_conv1_layer_call_and_return_conditional_losses_12170�
!quant_bn1/StatefulPartitionedCallStatefulPartitionedCall,quant_conv1/StatefulPartitionedCall:output:0quant_bn1_12616quant_bn1_12618quant_bn1_12620quant_bn1_12622*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_quant_bn1_layer_call_and_return_conditional_losses_12197�
#quant_relu1/StatefulPartitionedCallStatefulPartitionedCall*quant_bn1/StatefulPartitionedCall:output:0quant_relu1_12625quant_relu1_12627*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_relu1_layer_call_and_return_conditional_losses_12216�
quant_maxpool1/PartitionedCallPartitionedCall,quant_relu1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_maxpool1_layer_call_and_return_conditional_losses_12226�
#quant_conv2/StatefulPartitionedCallStatefulPartitionedCall'quant_maxpool1/PartitionedCall:output:0quant_conv2_12631quant_conv2_12633quant_conv2_12635quant_conv2_12637*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_conv2_layer_call_and_return_conditional_losses_12242�
!quant_bn2/StatefulPartitionedCallStatefulPartitionedCall,quant_conv2/StatefulPartitionedCall:output:0quant_bn2_12640quant_bn2_12642quant_bn2_12644quant_bn2_12646*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_quant_bn2_layer_call_and_return_conditional_losses_12269�
#quant_relu2/StatefulPartitionedCallStatefulPartitionedCall*quant_bn2/StatefulPartitionedCall:output:0quant_relu2_12649quant_relu2_12651*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_relu2_layer_call_and_return_conditional_losses_12288�
quant_maxpool2/PartitionedCallPartitionedCall,quant_relu2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_maxpool2_layer_call_and_return_conditional_losses_12298�
#quant_conv3/StatefulPartitionedCallStatefulPartitionedCall'quant_maxpool2/PartitionedCall:output:0quant_conv3_12655quant_conv3_12657quant_conv3_12659quant_conv3_12661*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_conv3_layer_call_and_return_conditional_losses_12314�
!quant_bn3/StatefulPartitionedCallStatefulPartitionedCall,quant_conv3/StatefulPartitionedCall:output:0quant_bn3_12664quant_bn3_12666quant_bn3_12668quant_bn3_12670*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_quant_bn3_layer_call_and_return_conditional_losses_12341�
#quant_relu3/StatefulPartitionedCallStatefulPartitionedCall*quant_bn3/StatefulPartitionedCall:output:0quant_relu3_12673quant_relu3_12675*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_relu3_layer_call_and_return_conditional_losses_12360�
quant_maxpool3/PartitionedCallPartitionedCall,quant_relu3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_maxpool3_layer_call_and_return_conditional_losses_12370�
quant_flatten/PartitionedCallPartitionedCall'quant_maxpool3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_quant_flatten_layer_call_and_return_conditional_losses_12377�
$quant_dense1/StatefulPartitionedCallStatefulPartitionedCall&quant_flatten/PartitionedCall:output:0quant_dense1_12680quant_dense1_12682quant_dense1_12684quant_dense1_12686quant_dense1_12688quant_dense1_12690*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_quant_dense1_layer_call_and_return_conditional_losses_12398�
quant_softmax/PartitionedCallPartitionedCall-quant_dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_quant_softmax_layer_call_and_return_conditional_losses_12416u
IdentityIdentity&quant_softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp"^quant_bn1/StatefulPartitionedCall"^quant_bn2/StatefulPartitionedCall"^quant_bn3/StatefulPartitionedCall$^quant_conv1/StatefulPartitionedCall$^quant_conv2/StatefulPartitionedCall$^quant_conv3/StatefulPartitionedCall%^quant_dense1/StatefulPartitionedCall$^quant_relu1/StatefulPartitionedCall$^quant_relu2/StatefulPartitionedCall$^quant_relu3/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!quant_bn1/StatefulPartitionedCall!quant_bn1/StatefulPartitionedCall2F
!quant_bn2/StatefulPartitionedCall!quant_bn2/StatefulPartitionedCall2F
!quant_bn3/StatefulPartitionedCall!quant_bn3/StatefulPartitionedCall2J
#quant_conv1/StatefulPartitionedCall#quant_conv1/StatefulPartitionedCall2J
#quant_conv2/StatefulPartitionedCall#quant_conv2/StatefulPartitionedCall2J
#quant_conv3/StatefulPartitionedCall#quant_conv3/StatefulPartitionedCall2L
$quant_dense1/StatefulPartitionedCall$quant_dense1/StatefulPartitionedCall2J
#quant_relu1/StatefulPartitionedCall#quant_relu1/StatefulPartitionedCall2J
#quant_relu2/StatefulPartitionedCall#quant_relu2/StatefulPartitionedCall2J
#quant_relu3/StatefulPartitionedCall#quant_relu3/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�,
�
F__inference_quant_conv3_layer_call_and_return_conditional_losses_11984

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2�
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:�
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:�
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:�
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:�
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0�
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(�
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
I
-__inference_quant_softmax_layer_call_fn_14677

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_quant_softmax_layer_call_and_return_conditional_losses_12416`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������
:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�N
�
G__inference_quant_dense1_layer_call_and_return_conditional_losses_12117

inputsB
/lastvaluequant_batchmin_readvariableop_resource:	�
/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:
@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1e
LastValueQuant/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/Const:output:0*
T0*
_output_shapes
: g
LastValueQuant/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/Const_1:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��{
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: |
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	�
*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       v
MovingAvgQuantize/BatchMinMinBiasAdd:output:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMaxMaxBiasAdd:output:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������
�
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
J
.__inference_quant_maxpool1_layer_call_fn_14125

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_maxpool1_layer_call_and_return_conditional_losses_12226h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  :W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
e
I__inference_quant_maxpool3_layer_call_and_return_conditional_losses_14538

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_quant_bn1_layer_call_and_return_conditional_losses_12197

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  :::::*
epsilon%o�:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������  �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������  : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
)__inference_MNIST_CNN_layer_call_fn_13430

inputs
unknown: 
	unknown_0: #
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9: 

unknown_10: $

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19: 

unknown_20: $

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29: 

unknown_30: 

unknown_31:	�


unknown_32: 

unknown_33: 

unknown_34:


unknown_35: 

unknown_36: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*0
_read_only_resource_inputs
!$*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_MNIST_CNN_layer_call_and_return_conditional_losses_12518o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_13349	
input
unknown: 
	unknown_0: #
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9: 

unknown_10: $

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19: 

unknown_20: $

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29: 

unknown_30: 

unknown_31:	�


unknown_32: 

unknown_33: 

unknown_34:


unknown_35: 

unknown_36: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_11490o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:���������  

_user_specified_nameinput
�
�
)__inference_quant_bn2_layer_call_fn_14229

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_quant_bn2_layer_call_and_return_conditional_losses_12269w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�,
�
F__inference_quant_conv1_layer_call_and_return_conditional_losses_13984

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2�
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:�
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:�
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:�
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:�
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0�
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(�
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������  �
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������  : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
F__inference_quant_conv1_layer_call_and_return_conditional_losses_12170

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0�
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(�
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������  �
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������  : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
>__inference_bn3_layer_call_and_return_conditional_losses_11661

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
d
H__inference_quant_softmax_layer_call_and_return_conditional_losses_12136

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:���������
Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������
:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
J
.__inference_quant_maxpool2_layer_call_fn_14324

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_maxpool2_layer_call_and_return_conditional_losses_11955h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�,
�
F__inference_quant_conv1_layer_call_and_return_conditional_losses_11776

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2�
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:�
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:�
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:�
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:�
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0�
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(�
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������  �
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������  : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
+__inference_quant_relu2_layer_call_fn_14274

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_relu2_layer_call_and_return_conditional_losses_11944w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_quant_bn1_layer_call_and_return_conditional_losses_11804

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  :::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������  �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������  : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
e
I__inference_quant_maxpool1_layer_call_and_return_conditional_losses_14130

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  :W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
+__inference_quant_conv1_layer_call_fn_13944

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_conv1_layer_call_and_return_conditional_losses_11776w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������  : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�N
�
G__inference_quant_dense1_layer_call_and_return_conditional_losses_14647

inputsB
/lastvaluequant_batchmin_readvariableop_resource:	�
/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:
@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1e
LastValueQuant/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/Const:output:0*
T0*
_output_shapes
: g
LastValueQuant/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/Const_1:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��{
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: |
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	�
*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       v
MovingAvgQuantize/BatchMinMinBiasAdd:output:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMaxMaxBiasAdd:output:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������
�
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
H__inference_quant_flatten_layer_call_and_return_conditional_losses_12377

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_quant_bn2_layer_call_and_return_conditional_losses_11908

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_quant_bn1_layer_call_and_return_conditional_losses_14043

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  :::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������  �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������  : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
+__inference_quant_conv3_layer_call_fn_14352

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_conv3_layer_call_and_return_conditional_losses_11984w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_quant_relu3_layer_call_and_return_conditional_losses_14523

inputsK
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1N
ReluReluinputs*
T0*/
_output_shapes
:����������
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
d
H__inference_quant_flatten_layer_call_and_return_conditional_losses_12067

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_quant_relu3_layer_call_fn_14487

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_relu3_layer_call_and_return_conditional_losses_12360w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_quant_conv1_layer_call_and_return_conditional_losses_13999

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0�
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(�
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������  �
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������  : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
#__inference_bn3_layer_call_fn_14844

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn3_layer_call_and_return_conditional_losses_11661�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
G__inference_quant_dense1_layer_call_and_return_conditional_losses_12398

inputsQ
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:	�
J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:
K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	�
*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������
�
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_quant_conv3_layer_call_fn_14365

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_conv3_layer_call_and_return_conditional_losses_12314w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
J
.__inference_quant_maxpool2_layer_call_fn_14329

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_maxpool2_layer_call_and_return_conditional_losses_12298h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
e
I__inference_quant_maxpool2_layer_call_and_return_conditional_losses_11955

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_quant_conv2_layer_call_and_return_conditional_losses_12242

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0�
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(�
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_quant_bn3_layer_call_and_return_conditional_losses_14469

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�*
�
F__inference_quant_relu1_layer_call_and_return_conditional_losses_11840

inputs@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1N
ReluReluinputs*
T0*/
_output_shapes
:���������  p
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:���������  �
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:���������  �
NoOpNoOp3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�*
�
F__inference_quant_relu1_layer_call_and_return_conditional_losses_14105

inputs@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1N
ReluReluinputs*
T0*/
_output_shapes
:���������  p
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:���������  �
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:���������  �
NoOpNoOp3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
)__inference_MNIST_CNN_layer_call_fn_12774	
input
unknown: 
	unknown_0: #
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9: 

unknown_10: $

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19: 

unknown_20: $

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29: 

unknown_30: 

unknown_31:	�


unknown_32: 

unknown_33: 

unknown_34:


unknown_35: 

unknown_36: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_MNIST_CNN_layer_call_and_return_conditional_losses_12695o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:���������  

_user_specified_nameinput
�
�
>__inference_bn3_layer_call_and_return_conditional_losses_14875

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
I
-__inference_quant_flatten_layer_call_fn_14548

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_quant_flatten_layer_call_and_return_conditional_losses_12067a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
)__inference_MNIST_CNN_layer_call_fn_13511

inputs
unknown: 
	unknown_0: #
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9: 

unknown_10: $

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19: 

unknown_20: $

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29: 

unknown_30: 

unknown_31:	�


unknown_32: 

unknown_33: 

unknown_34:


unknown_35: 

unknown_36: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_MNIST_CNN_layer_call_and_return_conditional_losses_12695o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
F__inference_quant_conv3_layer_call_and_return_conditional_losses_14407

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0�
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(�
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
#__inference_bn1_layer_call_fn_14700

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_11509�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
I
-__inference_quant_softmax_layer_call_fn_14672

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_quant_softmax_layer_call_and_return_conditional_losses_12136`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������
:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�#
�
I__inference_quantize_layer_layer_call_and_return_conditional_losses_11743

inputs;
1allvaluesquantize_minimum_readvariableop_resource: ;
1allvaluesquantize_maximum_readvariableop_resource: 
identity��#AllValuesQuantize/AssignMaxAllValue�#AllValuesQuantize/AssignMinAllValue�8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�(AllValuesQuantize/Maximum/ReadVariableOp�(AllValuesQuantize/Minimum/ReadVariableOpp
AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             l
AllValuesQuantize/BatchMinMininputs AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: r
AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             n
AllValuesQuantize/BatchMaxMaxinputs"AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: �
(AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype0�
AllValuesQuantize/MinimumMinimum0AllValuesQuantize/Minimum/ReadVariableOp:value:0#AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
AllValuesQuantize/Minimum_1MinimumAllValuesQuantize/Minimum:z:0&AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: �
(AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp1allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype0�
AllValuesQuantize/MaximumMaximum0AllValuesQuantize/Maximum/ReadVariableOp:value:0#AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
AllValuesQuantize/Maximum_1MaximumAllValuesQuantize/Maximum:z:0&AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: �
#AllValuesQuantize/AssignMinAllValueAssignVariableOp1allvaluesquantize_minimum_readvariableop_resourceAllValuesQuantize/Minimum_1:z:0)^AllValuesQuantize/Minimum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
#AllValuesQuantize/AssignMaxAllValueAssignVariableOp1allvaluesquantize_maximum_readvariableop_resourceAllValuesQuantize/Maximum_1:z:0)^AllValuesQuantize/Maximum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource$^AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype0�
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp1allvaluesquantize_maximum_readvariableop_resource$^AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype0�
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:���������  �
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:���������  �
NoOpNoOp$^AllValuesQuantize/AssignMaxAllValue$^AllValuesQuantize/AssignMinAllValue9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1)^AllValuesQuantize/Maximum/ReadVariableOp)^AllValuesQuantize/Minimum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 2J
#AllValuesQuantize/AssignMaxAllValue#AllValuesQuantize/AssignMaxAllValue2J
#AllValuesQuantize/AssignMinAllValue#AllValuesQuantize/AssignMinAllValue2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12T
(AllValuesQuantize/Maximum/ReadVariableOp(AllValuesQuantize/Maximum/ReadVariableOp2T
(AllValuesQuantize/Minimum/ReadVariableOp(AllValuesQuantize/Minimum/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
e
I__inference_quant_maxpool1_layer_call_and_return_conditional_losses_11851

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  :W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
��
�/
 __inference__wrapped_model_11490	
inputd
Zmnist_cnn_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: f
\mnist_cnn_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: x
^mnist_cnn_quant_conv1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:n
`mnist_cnn_quant_conv1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:n
`mnist_cnn_quant_conv1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:C
5mnist_cnn_quant_conv1_biasadd_readvariableop_resource:9
+mnist_cnn_quant_bn1_readvariableop_resource:;
-mnist_cnn_quant_bn1_readvariableop_1_resource:J
<mnist_cnn_quant_bn1_fusedbatchnormv3_readvariableop_resource:L
>mnist_cnn_quant_bn1_fusedbatchnormv3_readvariableop_1_resource:a
Wmnist_cnn_quant_relu1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: c
Ymnist_cnn_quant_relu1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: x
^mnist_cnn_quant_conv2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:n
`mnist_cnn_quant_conv2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:n
`mnist_cnn_quant_conv2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:C
5mnist_cnn_quant_conv2_biasadd_readvariableop_resource:9
+mnist_cnn_quant_bn2_readvariableop_resource:;
-mnist_cnn_quant_bn2_readvariableop_1_resource:J
<mnist_cnn_quant_bn2_fusedbatchnormv3_readvariableop_resource:L
>mnist_cnn_quant_bn2_fusedbatchnormv3_readvariableop_1_resource:a
Wmnist_cnn_quant_relu2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: c
Ymnist_cnn_quant_relu2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: x
^mnist_cnn_quant_conv3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:n
`mnist_cnn_quant_conv3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:n
`mnist_cnn_quant_conv3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:C
5mnist_cnn_quant_conv3_biasadd_readvariableop_resource:9
+mnist_cnn_quant_bn3_readvariableop_resource:;
-mnist_cnn_quant_bn3_readvariableop_1_resource:J
<mnist_cnn_quant_bn3_fusedbatchnormv3_readvariableop_resource:L
>mnist_cnn_quant_bn3_fusedbatchnormv3_readvariableop_1_resource:a
Wmnist_cnn_quant_relu3_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: c
Ymnist_cnn_quant_relu3_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: h
Umnist_cnn_quant_dense1_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:	�
a
Wmnist_cnn_quant_dense1_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: a
Wmnist_cnn_quant_dense1_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: D
6mnist_cnn_quant_dense1_biasadd_readvariableop_resource:
b
Xmnist_cnn_quant_dense1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: d
Zmnist_cnn_quant_dense1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��3MNIST_CNN/quant_bn1/FusedBatchNormV3/ReadVariableOp�5MNIST_CNN/quant_bn1/FusedBatchNormV3/ReadVariableOp_1�"MNIST_CNN/quant_bn1/ReadVariableOp�$MNIST_CNN/quant_bn1/ReadVariableOp_1�3MNIST_CNN/quant_bn2/FusedBatchNormV3/ReadVariableOp�5MNIST_CNN/quant_bn2/FusedBatchNormV3/ReadVariableOp_1�"MNIST_CNN/quant_bn2/ReadVariableOp�$MNIST_CNN/quant_bn2/ReadVariableOp_1�3MNIST_CNN/quant_bn3/FusedBatchNormV3/ReadVariableOp�5MNIST_CNN/quant_bn3/FusedBatchNormV3/ReadVariableOp_1�"MNIST_CNN/quant_bn3/ReadVariableOp�$MNIST_CNN/quant_bn3/ReadVariableOp_1�,MNIST_CNN/quant_conv1/BiasAdd/ReadVariableOp�UMNIST_CNN/quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp�WMNIST_CNN/quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1�WMNIST_CNN/quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2�,MNIST_CNN/quant_conv2/BiasAdd/ReadVariableOp�UMNIST_CNN/quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp�WMNIST_CNN/quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1�WMNIST_CNN/quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2�,MNIST_CNN/quant_conv3/BiasAdd/ReadVariableOp�UMNIST_CNN/quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp�WMNIST_CNN/quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1�WMNIST_CNN/quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2�-MNIST_CNN/quant_dense1/BiasAdd/ReadVariableOp�LMNIST_CNN/quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�NMNIST_CNN/quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�NMNIST_CNN/quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�OMNIST_CNN/quant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�QMNIST_CNN/quant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�NMNIST_CNN/quant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�PMNIST_CNN/quant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�NMNIST_CNN/quant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�PMNIST_CNN/quant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�NMNIST_CNN/quant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�PMNIST_CNN/quant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�QMNIST_CNN/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�SMNIST_CNN/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
QMNIST_CNN/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpZmnist_cnn_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
SMNIST_CNN/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp\mnist_cnn_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
BMNIST_CNN/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputYMNIST_CNN/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0[MNIST_CNN/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:���������  �
UMNIST_CNN/quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp^mnist_cnn_quant_conv1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0�
WMNIST_CNN/quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp`mnist_cnn_quant_conv1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0�
WMNIST_CNN/quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp`mnist_cnn_quant_conv1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0�
FMNIST_CNN/quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannel]MNIST_CNN/quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0_MNIST_CNN/quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0_MNIST_CNN/quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(�
MNIST_CNN/quant_conv1/Conv2DConv2DLMNIST_CNN/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0PMNIST_CNN/quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
�
,MNIST_CNN/quant_conv1/BiasAdd/ReadVariableOpReadVariableOp5mnist_cnn_quant_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
MNIST_CNN/quant_conv1/BiasAddBiasAdd%MNIST_CNN/quant_conv1/Conv2D:output:04MNIST_CNN/quant_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  �
"MNIST_CNN/quant_bn1/ReadVariableOpReadVariableOp+mnist_cnn_quant_bn1_readvariableop_resource*
_output_shapes
:*
dtype0�
$MNIST_CNN/quant_bn1/ReadVariableOp_1ReadVariableOp-mnist_cnn_quant_bn1_readvariableop_1_resource*
_output_shapes
:*
dtype0�
3MNIST_CNN/quant_bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp<mnist_cnn_quant_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
5MNIST_CNN/quant_bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>mnist_cnn_quant_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
$MNIST_CNN/quant_bn1/FusedBatchNormV3FusedBatchNormV3&MNIST_CNN/quant_conv1/BiasAdd:output:0*MNIST_CNN/quant_bn1/ReadVariableOp:value:0,MNIST_CNN/quant_bn1/ReadVariableOp_1:value:0;MNIST_CNN/quant_bn1/FusedBatchNormV3/ReadVariableOp:value:0=MNIST_CNN/quant_bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  :::::*
epsilon%o�:*
is_training( �
MNIST_CNN/quant_relu1/ReluRelu(MNIST_CNN/quant_bn1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������  �
NMNIST_CNN/quant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpWmnist_cnn_quant_relu1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
PMNIST_CNN/quant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpYmnist_cnn_quant_relu1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
?MNIST_CNN/quant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars(MNIST_CNN/quant_relu1/Relu:activations:0VMNIST_CNN/quant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0XMNIST_CNN/quant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:���������  �
 MNIST_CNN/quant_maxpool1/MaxPoolMaxPoolIMNIST_CNN/quant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
�
UMNIST_CNN/quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp^mnist_cnn_quant_conv2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0�
WMNIST_CNN/quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp`mnist_cnn_quant_conv2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0�
WMNIST_CNN/quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp`mnist_cnn_quant_conv2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0�
FMNIST_CNN/quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannel]MNIST_CNN/quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0_MNIST_CNN/quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0_MNIST_CNN/quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(�
MNIST_CNN/quant_conv2/Conv2DConv2D)MNIST_CNN/quant_maxpool1/MaxPool:output:0PMNIST_CNN/quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
,MNIST_CNN/quant_conv2/BiasAdd/ReadVariableOpReadVariableOp5mnist_cnn_quant_conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
MNIST_CNN/quant_conv2/BiasAddBiasAdd%MNIST_CNN/quant_conv2/Conv2D:output:04MNIST_CNN/quant_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
"MNIST_CNN/quant_bn2/ReadVariableOpReadVariableOp+mnist_cnn_quant_bn2_readvariableop_resource*
_output_shapes
:*
dtype0�
$MNIST_CNN/quant_bn2/ReadVariableOp_1ReadVariableOp-mnist_cnn_quant_bn2_readvariableop_1_resource*
_output_shapes
:*
dtype0�
3MNIST_CNN/quant_bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp<mnist_cnn_quant_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
5MNIST_CNN/quant_bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>mnist_cnn_quant_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
$MNIST_CNN/quant_bn2/FusedBatchNormV3FusedBatchNormV3&MNIST_CNN/quant_conv2/BiasAdd:output:0*MNIST_CNN/quant_bn2/ReadVariableOp:value:0,MNIST_CNN/quant_bn2/ReadVariableOp_1:value:0;MNIST_CNN/quant_bn2/FusedBatchNormV3/ReadVariableOp:value:0=MNIST_CNN/quant_bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( �
MNIST_CNN/quant_relu2/ReluRelu(MNIST_CNN/quant_bn2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:����������
NMNIST_CNN/quant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpWmnist_cnn_quant_relu2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
PMNIST_CNN/quant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpYmnist_cnn_quant_relu2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
?MNIST_CNN/quant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars(MNIST_CNN/quant_relu2/Relu:activations:0VMNIST_CNN/quant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0XMNIST_CNN/quant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:����������
 MNIST_CNN/quant_maxpool2/MaxPoolMaxPoolIMNIST_CNN/quant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
�
UMNIST_CNN/quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp^mnist_cnn_quant_conv3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0�
WMNIST_CNN/quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp`mnist_cnn_quant_conv3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0�
WMNIST_CNN/quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp`mnist_cnn_quant_conv3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0�
FMNIST_CNN/quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannel]MNIST_CNN/quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0_MNIST_CNN/quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0_MNIST_CNN/quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(�
MNIST_CNN/quant_conv3/Conv2DConv2D)MNIST_CNN/quant_maxpool2/MaxPool:output:0PMNIST_CNN/quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
,MNIST_CNN/quant_conv3/BiasAdd/ReadVariableOpReadVariableOp5mnist_cnn_quant_conv3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
MNIST_CNN/quant_conv3/BiasAddBiasAdd%MNIST_CNN/quant_conv3/Conv2D:output:04MNIST_CNN/quant_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
"MNIST_CNN/quant_bn3/ReadVariableOpReadVariableOp+mnist_cnn_quant_bn3_readvariableop_resource*
_output_shapes
:*
dtype0�
$MNIST_CNN/quant_bn3/ReadVariableOp_1ReadVariableOp-mnist_cnn_quant_bn3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
3MNIST_CNN/quant_bn3/FusedBatchNormV3/ReadVariableOpReadVariableOp<mnist_cnn_quant_bn3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
5MNIST_CNN/quant_bn3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>mnist_cnn_quant_bn3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
$MNIST_CNN/quant_bn3/FusedBatchNormV3FusedBatchNormV3&MNIST_CNN/quant_conv3/BiasAdd:output:0*MNIST_CNN/quant_bn3/ReadVariableOp:value:0,MNIST_CNN/quant_bn3/ReadVariableOp_1:value:0;MNIST_CNN/quant_bn3/FusedBatchNormV3/ReadVariableOp:value:0=MNIST_CNN/quant_bn3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( �
MNIST_CNN/quant_relu3/ReluRelu(MNIST_CNN/quant_bn3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:����������
NMNIST_CNN/quant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpWmnist_cnn_quant_relu3_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
PMNIST_CNN/quant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpYmnist_cnn_quant_relu3_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
?MNIST_CNN/quant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars(MNIST_CNN/quant_relu3/Relu:activations:0VMNIST_CNN/quant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0XMNIST_CNN/quant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:����������
 MNIST_CNN/quant_maxpool3/MaxPoolMaxPoolIMNIST_CNN/quant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
n
MNIST_CNN/quant_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
MNIST_CNN/quant_flatten/ReshapeReshape)MNIST_CNN/quant_maxpool3/MaxPool:output:0&MNIST_CNN/quant_flatten/Const:output:0*
T0*(
_output_shapes
:�����������
LMNIST_CNN/quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpUmnist_cnn_quant_dense1_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
NMNIST_CNN/quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpWmnist_cnn_quant_dense1_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
NMNIST_CNN/quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpWmnist_cnn_quant_dense1_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
=MNIST_CNN/quant_dense1/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsTMNIST_CNN/quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0VMNIST_CNN/quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0VMNIST_CNN/quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	�
*
narrow_range(�
MNIST_CNN/quant_dense1/MatMulMatMul(MNIST_CNN/quant_flatten/Reshape:output:0GMNIST_CNN/quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������
�
-MNIST_CNN/quant_dense1/BiasAdd/ReadVariableOpReadVariableOp6mnist_cnn_quant_dense1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
MNIST_CNN/quant_dense1/BiasAddBiasAdd'MNIST_CNN/quant_dense1/MatMul:product:05MNIST_CNN/quant_dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
OMNIST_CNN/quant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpXmnist_cnn_quant_dense1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
QMNIST_CNN/quant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpZmnist_cnn_quant_dense1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
@MNIST_CNN/quant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars'MNIST_CNN/quant_dense1/BiasAdd:output:0WMNIST_CNN/quant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0YMNIST_CNN/quant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������
�
MNIST_CNN/quant_softmax/SoftmaxSoftmaxJMNIST_CNN/quant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������
x
IdentityIdentity)MNIST_CNN/quant_softmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp4^MNIST_CNN/quant_bn1/FusedBatchNormV3/ReadVariableOp6^MNIST_CNN/quant_bn1/FusedBatchNormV3/ReadVariableOp_1#^MNIST_CNN/quant_bn1/ReadVariableOp%^MNIST_CNN/quant_bn1/ReadVariableOp_14^MNIST_CNN/quant_bn2/FusedBatchNormV3/ReadVariableOp6^MNIST_CNN/quant_bn2/FusedBatchNormV3/ReadVariableOp_1#^MNIST_CNN/quant_bn2/ReadVariableOp%^MNIST_CNN/quant_bn2/ReadVariableOp_14^MNIST_CNN/quant_bn3/FusedBatchNormV3/ReadVariableOp6^MNIST_CNN/quant_bn3/FusedBatchNormV3/ReadVariableOp_1#^MNIST_CNN/quant_bn3/ReadVariableOp%^MNIST_CNN/quant_bn3/ReadVariableOp_1-^MNIST_CNN/quant_conv1/BiasAdd/ReadVariableOpV^MNIST_CNN/quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpX^MNIST_CNN/quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1X^MNIST_CNN/quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2-^MNIST_CNN/quant_conv2/BiasAdd/ReadVariableOpV^MNIST_CNN/quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpX^MNIST_CNN/quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1X^MNIST_CNN/quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2-^MNIST_CNN/quant_conv3/BiasAdd/ReadVariableOpV^MNIST_CNN/quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpX^MNIST_CNN/quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1X^MNIST_CNN/quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2.^MNIST_CNN/quant_dense1/BiasAdd/ReadVariableOpM^MNIST_CNN/quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpO^MNIST_CNN/quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1O^MNIST_CNN/quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2P^MNIST_CNN/quant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpR^MNIST_CNN/quant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1O^MNIST_CNN/quant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpQ^MNIST_CNN/quant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1O^MNIST_CNN/quant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpQ^MNIST_CNN/quant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1O^MNIST_CNN/quant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpQ^MNIST_CNN/quant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1R^MNIST_CNN/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpT^MNIST_CNN/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
3MNIST_CNN/quant_bn1/FusedBatchNormV3/ReadVariableOp3MNIST_CNN/quant_bn1/FusedBatchNormV3/ReadVariableOp2n
5MNIST_CNN/quant_bn1/FusedBatchNormV3/ReadVariableOp_15MNIST_CNN/quant_bn1/FusedBatchNormV3/ReadVariableOp_12H
"MNIST_CNN/quant_bn1/ReadVariableOp"MNIST_CNN/quant_bn1/ReadVariableOp2L
$MNIST_CNN/quant_bn1/ReadVariableOp_1$MNIST_CNN/quant_bn1/ReadVariableOp_12j
3MNIST_CNN/quant_bn2/FusedBatchNormV3/ReadVariableOp3MNIST_CNN/quant_bn2/FusedBatchNormV3/ReadVariableOp2n
5MNIST_CNN/quant_bn2/FusedBatchNormV3/ReadVariableOp_15MNIST_CNN/quant_bn2/FusedBatchNormV3/ReadVariableOp_12H
"MNIST_CNN/quant_bn2/ReadVariableOp"MNIST_CNN/quant_bn2/ReadVariableOp2L
$MNIST_CNN/quant_bn2/ReadVariableOp_1$MNIST_CNN/quant_bn2/ReadVariableOp_12j
3MNIST_CNN/quant_bn3/FusedBatchNormV3/ReadVariableOp3MNIST_CNN/quant_bn3/FusedBatchNormV3/ReadVariableOp2n
5MNIST_CNN/quant_bn3/FusedBatchNormV3/ReadVariableOp_15MNIST_CNN/quant_bn3/FusedBatchNormV3/ReadVariableOp_12H
"MNIST_CNN/quant_bn3/ReadVariableOp"MNIST_CNN/quant_bn3/ReadVariableOp2L
$MNIST_CNN/quant_bn3/ReadVariableOp_1$MNIST_CNN/quant_bn3/ReadVariableOp_12\
,MNIST_CNN/quant_conv1/BiasAdd/ReadVariableOp,MNIST_CNN/quant_conv1/BiasAdd/ReadVariableOp2�
UMNIST_CNN/quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpUMNIST_CNN/quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2�
WMNIST_CNN/quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1WMNIST_CNN/quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12�
WMNIST_CNN/quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2WMNIST_CNN/quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22\
,MNIST_CNN/quant_conv2/BiasAdd/ReadVariableOp,MNIST_CNN/quant_conv2/BiasAdd/ReadVariableOp2�
UMNIST_CNN/quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpUMNIST_CNN/quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2�
WMNIST_CNN/quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1WMNIST_CNN/quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12�
WMNIST_CNN/quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2WMNIST_CNN/quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22\
,MNIST_CNN/quant_conv3/BiasAdd/ReadVariableOp,MNIST_CNN/quant_conv3/BiasAdd/ReadVariableOp2�
UMNIST_CNN/quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpUMNIST_CNN/quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2�
WMNIST_CNN/quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1WMNIST_CNN/quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12�
WMNIST_CNN/quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2WMNIST_CNN/quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22^
-MNIST_CNN/quant_dense1/BiasAdd/ReadVariableOp-MNIST_CNN/quant_dense1/BiasAdd/ReadVariableOp2�
LMNIST_CNN/quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpLMNIST_CNN/quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
NMNIST_CNN/quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1NMNIST_CNN/quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
NMNIST_CNN/quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2NMNIST_CNN/quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
OMNIST_CNN/quant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpOMNIST_CNN/quant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
QMNIST_CNN/quant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1QMNIST_CNN/quant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
NMNIST_CNN/quant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpNMNIST_CNN/quant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
PMNIST_CNN/quant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1PMNIST_CNN/quant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
NMNIST_CNN/quant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpNMNIST_CNN/quant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
PMNIST_CNN/quant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1PMNIST_CNN/quant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
NMNIST_CNN/quant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpNMNIST_CNN/quant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
PMNIST_CNN/quant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1PMNIST_CNN/quant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
QMNIST_CNN/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpQMNIST_CNN/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
SMNIST_CNN/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1SMNIST_CNN/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:V R
/
_output_shapes
:���������  

_user_specified_nameinput
��
�*
D__inference_MNIST_CNN_layer_call_and_return_conditional_losses_13883

inputsZ
Pquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: n
Tquant_conv1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:d
Vquant_conv1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:d
Vquant_conv1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:9
+quant_conv1_biasadd_readvariableop_resource:/
!quant_bn1_readvariableop_resource:1
#quant_bn1_readvariableop_1_resource:@
2quant_bn1_fusedbatchnormv3_readvariableop_resource:B
4quant_bn1_fusedbatchnormv3_readvariableop_1_resource:W
Mquant_relu1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: Y
Oquant_relu1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: n
Tquant_conv2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:d
Vquant_conv2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:d
Vquant_conv2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:9
+quant_conv2_biasadd_readvariableop_resource:/
!quant_bn2_readvariableop_resource:1
#quant_bn2_readvariableop_1_resource:@
2quant_bn2_fusedbatchnormv3_readvariableop_resource:B
4quant_bn2_fusedbatchnormv3_readvariableop_1_resource:W
Mquant_relu2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: Y
Oquant_relu2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: n
Tquant_conv3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:d
Vquant_conv3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:d
Vquant_conv3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:9
+quant_conv3_biasadd_readvariableop_resource:/
!quant_bn3_readvariableop_resource:1
#quant_bn3_readvariableop_1_resource:@
2quant_bn3_fusedbatchnormv3_readvariableop_resource:B
4quant_bn3_fusedbatchnormv3_readvariableop_1_resource:W
Mquant_relu3_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: Y
Oquant_relu3_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: ^
Kquant_dense1_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:	�
W
Mquant_dense1_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: W
Mquant_dense1_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: :
,quant_dense1_biasadd_readvariableop_resource:
X
Nquant_dense1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: Z
Pquant_dense1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��)quant_bn1/FusedBatchNormV3/ReadVariableOp�+quant_bn1/FusedBatchNormV3/ReadVariableOp_1�quant_bn1/ReadVariableOp�quant_bn1/ReadVariableOp_1�)quant_bn2/FusedBatchNormV3/ReadVariableOp�+quant_bn2/FusedBatchNormV3/ReadVariableOp_1�quant_bn2/ReadVariableOp�quant_bn2/ReadVariableOp_1�)quant_bn3/FusedBatchNormV3/ReadVariableOp�+quant_bn3/FusedBatchNormV3/ReadVariableOp_1�quant_bn3/ReadVariableOp�quant_bn3/ReadVariableOp_1�"quant_conv1/BiasAdd/ReadVariableOp�Kquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp�Mquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1�Mquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2�"quant_conv2/BiasAdd/ReadVariableOp�Kquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp�Mquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1�Mquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2�"quant_conv3/BiasAdd/ReadVariableOp�Kquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp�Mquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1�Mquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2�#quant_dense1/BiasAdd/ReadVariableOp�Bquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Dquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Dquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Equant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Gquant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Dquant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Dquant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Dquant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
8quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsOquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:���������  �
Kquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpTquant_conv1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0�
Mquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpVquant_conv1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0�
Mquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpVquant_conv1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0�
<quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelSquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Uquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Uquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(�
quant_conv1/Conv2DConv2DBquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Fquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
�
"quant_conv1/BiasAdd/ReadVariableOpReadVariableOp+quant_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_conv1/BiasAddBiasAddquant_conv1/Conv2D:output:0*quant_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  v
quant_bn1/ReadVariableOpReadVariableOp!quant_bn1_readvariableop_resource*
_output_shapes
:*
dtype0z
quant_bn1/ReadVariableOp_1ReadVariableOp#quant_bn1_readvariableop_1_resource*
_output_shapes
:*
dtype0�
)quant_bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp2quant_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
+quant_bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp4quant_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
quant_bn1/FusedBatchNormV3FusedBatchNormV3quant_conv1/BiasAdd:output:0 quant_bn1/ReadVariableOp:value:0"quant_bn1/ReadVariableOp_1:value:01quant_bn1/FusedBatchNormV3/ReadVariableOp:value:03quant_bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  :::::*
epsilon%o�:*
is_training( r
quant_relu1/ReluReluquant_bn1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������  �
Dquant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpMquant_relu1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Fquant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpOquant_relu1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
5quant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_relu1/Relu:activations:0Lquant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:���������  �
quant_maxpool1/MaxPoolMaxPool?quant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
�
Kquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpTquant_conv2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0�
Mquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpVquant_conv2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0�
Mquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpVquant_conv2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0�
<quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelSquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Uquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Uquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(�
quant_conv2/Conv2DConv2Dquant_maxpool1/MaxPool:output:0Fquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
"quant_conv2/BiasAdd/ReadVariableOpReadVariableOp+quant_conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_conv2/BiasAddBiasAddquant_conv2/Conv2D:output:0*quant_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v
quant_bn2/ReadVariableOpReadVariableOp!quant_bn2_readvariableop_resource*
_output_shapes
:*
dtype0z
quant_bn2/ReadVariableOp_1ReadVariableOp#quant_bn2_readvariableop_1_resource*
_output_shapes
:*
dtype0�
)quant_bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp2quant_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
+quant_bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp4quant_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
quant_bn2/FusedBatchNormV3FusedBatchNormV3quant_conv2/BiasAdd:output:0 quant_bn2/ReadVariableOp:value:0"quant_bn2/ReadVariableOp_1:value:01quant_bn2/FusedBatchNormV3/ReadVariableOp:value:03quant_bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( r
quant_relu2/ReluReluquant_bn2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:����������
Dquant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpMquant_relu2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Fquant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpOquant_relu2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
5quant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_relu2/Relu:activations:0Lquant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:����������
quant_maxpool2/MaxPoolMaxPool?quant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
�
Kquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpTquant_conv3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0�
Mquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpVquant_conv3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0�
Mquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpVquant_conv3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0�
<quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelSquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Uquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Uquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(�
quant_conv3/Conv2DConv2Dquant_maxpool2/MaxPool:output:0Fquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
"quant_conv3/BiasAdd/ReadVariableOpReadVariableOp+quant_conv3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_conv3/BiasAddBiasAddquant_conv3/Conv2D:output:0*quant_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v
quant_bn3/ReadVariableOpReadVariableOp!quant_bn3_readvariableop_resource*
_output_shapes
:*
dtype0z
quant_bn3/ReadVariableOp_1ReadVariableOp#quant_bn3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
)quant_bn3/FusedBatchNormV3/ReadVariableOpReadVariableOp2quant_bn3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
+quant_bn3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp4quant_bn3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
quant_bn3/FusedBatchNormV3FusedBatchNormV3quant_conv3/BiasAdd:output:0 quant_bn3/ReadVariableOp:value:0"quant_bn3/ReadVariableOp_1:value:01quant_bn3/FusedBatchNormV3/ReadVariableOp:value:03quant_bn3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( r
quant_relu3/ReluReluquant_bn3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:����������
Dquant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpMquant_relu3_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Fquant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpOquant_relu3_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
5quant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_relu3/Relu:activations:0Lquant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:����������
quant_maxpool3/MaxPoolMaxPool?quant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
d
quant_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
quant_flatten/ReshapeReshapequant_maxpool3/MaxPool:output:0quant_flatten/Const:output:0*
T0*(
_output_shapes
:�����������
Bquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpKquant_dense1_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
Dquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpMquant_dense1_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Dquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpMquant_dense1_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
3quant_dense1/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsJquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Lquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Lquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	�
*
narrow_range(�
quant_dense1/MatMulMatMulquant_flatten/Reshape:output:0=quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������
�
#quant_dense1/BiasAdd/ReadVariableOpReadVariableOp,quant_dense1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
quant_dense1/BiasAddBiasAddquant_dense1/MatMul:product:0+quant_dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
Equant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpNquant_dense1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Gquant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpPquant_dense1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
6quant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_dense1/BiasAdd:output:0Mquant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Oquant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������
�
quant_softmax/SoftmaxSoftmax@quant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������
n
IdentityIdentityquant_softmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp*^quant_bn1/FusedBatchNormV3/ReadVariableOp,^quant_bn1/FusedBatchNormV3/ReadVariableOp_1^quant_bn1/ReadVariableOp^quant_bn1/ReadVariableOp_1*^quant_bn2/FusedBatchNormV3/ReadVariableOp,^quant_bn2/FusedBatchNormV3/ReadVariableOp_1^quant_bn2/ReadVariableOp^quant_bn2/ReadVariableOp_1*^quant_bn3/FusedBatchNormV3/ReadVariableOp,^quant_bn3/FusedBatchNormV3/ReadVariableOp_1^quant_bn3/ReadVariableOp^quant_bn3/ReadVariableOp_1#^quant_conv1/BiasAdd/ReadVariableOpL^quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpN^quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1N^quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2#^quant_conv2/BiasAdd/ReadVariableOpL^quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpN^quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1N^quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2#^quant_conv3/BiasAdd/ReadVariableOpL^quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpN^quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1N^quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2$^quant_dense1/BiasAdd/ReadVariableOpC^quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpE^quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1E^quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2F^quant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpH^quant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1E^quant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1E^quant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1E^quant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1H^quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2V
)quant_bn1/FusedBatchNormV3/ReadVariableOp)quant_bn1/FusedBatchNormV3/ReadVariableOp2Z
+quant_bn1/FusedBatchNormV3/ReadVariableOp_1+quant_bn1/FusedBatchNormV3/ReadVariableOp_124
quant_bn1/ReadVariableOpquant_bn1/ReadVariableOp28
quant_bn1/ReadVariableOp_1quant_bn1/ReadVariableOp_12V
)quant_bn2/FusedBatchNormV3/ReadVariableOp)quant_bn2/FusedBatchNormV3/ReadVariableOp2Z
+quant_bn2/FusedBatchNormV3/ReadVariableOp_1+quant_bn2/FusedBatchNormV3/ReadVariableOp_124
quant_bn2/ReadVariableOpquant_bn2/ReadVariableOp28
quant_bn2/ReadVariableOp_1quant_bn2/ReadVariableOp_12V
)quant_bn3/FusedBatchNormV3/ReadVariableOp)quant_bn3/FusedBatchNormV3/ReadVariableOp2Z
+quant_bn3/FusedBatchNormV3/ReadVariableOp_1+quant_bn3/FusedBatchNormV3/ReadVariableOp_124
quant_bn3/ReadVariableOpquant_bn3/ReadVariableOp28
quant_bn3/ReadVariableOp_1quant_bn3/ReadVariableOp_12H
"quant_conv1/BiasAdd/ReadVariableOp"quant_conv1/BiasAdd/ReadVariableOp2�
Kquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpKquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2�
Mquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Mquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12�
Mquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Mquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22H
"quant_conv2/BiasAdd/ReadVariableOp"quant_conv2/BiasAdd/ReadVariableOp2�
Kquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpKquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2�
Mquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Mquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12�
Mquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Mquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22H
"quant_conv3/BiasAdd/ReadVariableOp"quant_conv3/BiasAdd/ReadVariableOp2�
Kquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpKquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2�
Mquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Mquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12�
Mquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Mquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22J
#quant_dense1/BiasAdd/ReadVariableOp#quant_dense1/BiasAdd/ReadVariableOp2�
Bquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpBquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Dquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Dquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Dquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Dquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Equant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpEquant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Gquant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Gquant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Dquant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpDquant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Dquant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpDquant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Dquant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpDquant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
J
.__inference_quant_maxpool1_layer_call_fn_14120

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_maxpool1_layer_call_and_return_conditional_losses_11851h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  :W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�,
�
F__inference_quant_conv2_layer_call_and_return_conditional_losses_11880

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2�
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:�
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:�
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:�
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:�
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0�
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(�
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
_
C__inference_maxpool2_layer_call_and_return_conditional_losses_11636

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
d
H__inference_quant_flatten_layer_call_and_return_conditional_losses_14565

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
I
-__inference_quant_flatten_layer_call_fn_14553

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_quant_flatten_layer_call_and_return_conditional_losses_12377a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_quantize_layer_layer_call_and_return_conditional_losses_12150

inputsK
Aallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: M
Callvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCallvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:���������  �
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:���������  �
NoOpNoOp9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
��
�4
!__inference__traced_restore_15728
file_prefix<
2assignvariableop_quantize_layer_quantize_layer_min: >
4assignvariableop_1_quantize_layer_quantize_layer_max: :
0assignvariableop_2_quantize_layer_optimizer_step: 7
-assignvariableop_3_quant_conv1_optimizer_step: 7
)assignvariableop_4_quant_conv1_kernel_min:7
)assignvariableop_5_quant_conv1_kernel_max:5
+assignvariableop_6_quant_bn1_optimizer_step: 7
-assignvariableop_7_quant_relu1_optimizer_step: 3
)assignvariableop_8_quant_relu1_output_min: 3
)assignvariableop_9_quant_relu1_output_max: ;
1assignvariableop_10_quant_maxpool1_optimizer_step: 8
.assignvariableop_11_quant_conv2_optimizer_step: 8
*assignvariableop_12_quant_conv2_kernel_min:8
*assignvariableop_13_quant_conv2_kernel_max:6
,assignvariableop_14_quant_bn2_optimizer_step: 8
.assignvariableop_15_quant_relu2_optimizer_step: 4
*assignvariableop_16_quant_relu2_output_min: 4
*assignvariableop_17_quant_relu2_output_max: ;
1assignvariableop_18_quant_maxpool2_optimizer_step: 8
.assignvariableop_19_quant_conv3_optimizer_step: 8
*assignvariableop_20_quant_conv3_kernel_min:8
*assignvariableop_21_quant_conv3_kernel_max:6
,assignvariableop_22_quant_bn3_optimizer_step: 8
.assignvariableop_23_quant_relu3_optimizer_step: 4
*assignvariableop_24_quant_relu3_output_min: 4
*assignvariableop_25_quant_relu3_output_max: ;
1assignvariableop_26_quant_maxpool3_optimizer_step: :
0assignvariableop_27_quant_flatten_optimizer_step: 9
/assignvariableop_28_quant_dense1_optimizer_step: 5
+assignvariableop_29_quant_dense1_kernel_min: 5
+assignvariableop_30_quant_dense1_kernel_max: >
4assignvariableop_31_quant_dense1_post_activation_min: >
4assignvariableop_32_quant_dense1_post_activation_max: :
0assignvariableop_33_quant_softmax_optimizer_step: :
 assignvariableop_34_conv1_kernel:,
assignvariableop_35_conv1_bias:+
assignvariableop_36_bn1_gamma:*
assignvariableop_37_bn1_beta:1
#assignvariableop_38_bn1_moving_mean:5
'assignvariableop_39_bn1_moving_variance::
 assignvariableop_40_conv2_kernel:,
assignvariableop_41_conv2_bias:+
assignvariableop_42_bn2_gamma:*
assignvariableop_43_bn2_beta:1
#assignvariableop_44_bn2_moving_mean:5
'assignvariableop_45_bn2_moving_variance::
 assignvariableop_46_conv3_kernel:,
assignvariableop_47_conv3_bias:+
assignvariableop_48_bn3_gamma:*
assignvariableop_49_bn3_beta:1
#assignvariableop_50_bn3_moving_mean:5
'assignvariableop_51_bn3_moving_variance:4
!assignvariableop_52_dense1_kernel:	�
-
assignvariableop_53_dense1_bias:
'
assignvariableop_54_iteration:	 +
!assignvariableop_55_learning_rate: A
'assignvariableop_56_adam_m_conv1_kernel:A
'assignvariableop_57_adam_v_conv1_kernel:3
%assignvariableop_58_adam_m_conv1_bias:3
%assignvariableop_59_adam_v_conv1_bias:2
$assignvariableop_60_adam_m_bn1_gamma:2
$assignvariableop_61_adam_v_bn1_gamma:1
#assignvariableop_62_adam_m_bn1_beta:1
#assignvariableop_63_adam_v_bn1_beta:A
'assignvariableop_64_adam_m_conv2_kernel:A
'assignvariableop_65_adam_v_conv2_kernel:3
%assignvariableop_66_adam_m_conv2_bias:3
%assignvariableop_67_adam_v_conv2_bias:2
$assignvariableop_68_adam_m_bn2_gamma:2
$assignvariableop_69_adam_v_bn2_gamma:1
#assignvariableop_70_adam_m_bn2_beta:1
#assignvariableop_71_adam_v_bn2_beta:A
'assignvariableop_72_adam_m_conv3_kernel:A
'assignvariableop_73_adam_v_conv3_kernel:3
%assignvariableop_74_adam_m_conv3_bias:3
%assignvariableop_75_adam_v_conv3_bias:2
$assignvariableop_76_adam_m_bn3_gamma:2
$assignvariableop_77_adam_v_bn3_gamma:1
#assignvariableop_78_adam_m_bn3_beta:1
#assignvariableop_79_adam_v_bn3_beta:;
(assignvariableop_80_adam_m_dense1_kernel:	�
;
(assignvariableop_81_adam_v_dense1_kernel:	�
4
&assignvariableop_82_adam_m_dense1_bias:
4
&assignvariableop_83_adam_v_dense1_bias:
%
assignvariableop_84_total_1: %
assignvariableop_85_count_1: #
assignvariableop_86_total: #
assignvariableop_87_count: 
identity_89��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_9�&
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*�%
value�%B�%YBBlayer_with_weights-0/quantize_layer_min/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/quantize_layer_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/output_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/output_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-6/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-7/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/output_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/output_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-8/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-9/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-9/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-9/kernel_max/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-10/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-11/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-11/output_min/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-11/output_max/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-12/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-13/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-14/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-14/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-14/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-14/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-14/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-15/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*�
value�B�YB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*g
dtypes]
[2Y	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp2assignvariableop_quantize_layer_quantize_layer_minIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp4assignvariableop_1_quantize_layer_quantize_layer_maxIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp0assignvariableop_2_quantize_layer_optimizer_stepIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp-assignvariableop_3_quant_conv1_optimizer_stepIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp)assignvariableop_4_quant_conv1_kernel_minIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp)assignvariableop_5_quant_conv1_kernel_maxIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp+assignvariableop_6_quant_bn1_optimizer_stepIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp-assignvariableop_7_quant_relu1_optimizer_stepIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp)assignvariableop_8_quant_relu1_output_minIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp)assignvariableop_9_quant_relu1_output_maxIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp1assignvariableop_10_quant_maxpool1_optimizer_stepIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp.assignvariableop_11_quant_conv2_optimizer_stepIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp*assignvariableop_12_quant_conv2_kernel_minIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp*assignvariableop_13_quant_conv2_kernel_maxIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp,assignvariableop_14_quant_bn2_optimizer_stepIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp.assignvariableop_15_quant_relu2_optimizer_stepIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp*assignvariableop_16_quant_relu2_output_minIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp*assignvariableop_17_quant_relu2_output_maxIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp1assignvariableop_18_quant_maxpool2_optimizer_stepIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp.assignvariableop_19_quant_conv3_optimizer_stepIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp*assignvariableop_20_quant_conv3_kernel_minIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp*assignvariableop_21_quant_conv3_kernel_maxIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp,assignvariableop_22_quant_bn3_optimizer_stepIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp.assignvariableop_23_quant_relu3_optimizer_stepIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp*assignvariableop_24_quant_relu3_output_minIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp*assignvariableop_25_quant_relu3_output_maxIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp1assignvariableop_26_quant_maxpool3_optimizer_stepIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp0assignvariableop_27_quant_flatten_optimizer_stepIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp/assignvariableop_28_quant_dense1_optimizer_stepIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp+assignvariableop_29_quant_dense1_kernel_minIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp+assignvariableop_30_quant_dense1_kernel_maxIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp4assignvariableop_31_quant_dense1_post_activation_minIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp4assignvariableop_32_quant_dense1_post_activation_maxIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp0assignvariableop_33_quant_softmax_optimizer_stepIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp assignvariableop_34_conv1_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOpassignvariableop_35_conv1_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpassignvariableop_36_bn1_gammaIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpassignvariableop_37_bn1_betaIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp#assignvariableop_38_bn1_moving_meanIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp'assignvariableop_39_bn1_moving_varianceIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp assignvariableop_40_conv2_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpassignvariableop_41_conv2_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOpassignvariableop_42_bn2_gammaIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOpassignvariableop_43_bn2_betaIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp#assignvariableop_44_bn2_moving_meanIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp'assignvariableop_45_bn2_moving_varianceIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp assignvariableop_46_conv3_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOpassignvariableop_47_conv3_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOpassignvariableop_48_bn3_gammaIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOpassignvariableop_49_bn3_betaIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp#assignvariableop_50_bn3_moving_meanIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp'assignvariableop_51_bn3_moving_varianceIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp!assignvariableop_52_dense1_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOpassignvariableop_53_dense1_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_54AssignVariableOpassignvariableop_54_iterationIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp!assignvariableop_55_learning_rateIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp'assignvariableop_56_adam_m_conv1_kernelIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp'assignvariableop_57_adam_v_conv1_kernelIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp%assignvariableop_58_adam_m_conv1_biasIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp%assignvariableop_59_adam_v_conv1_biasIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp$assignvariableop_60_adam_m_bn1_gammaIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp$assignvariableop_61_adam_v_bn1_gammaIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp#assignvariableop_62_adam_m_bn1_betaIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp#assignvariableop_63_adam_v_bn1_betaIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp'assignvariableop_64_adam_m_conv2_kernelIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp'assignvariableop_65_adam_v_conv2_kernelIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp%assignvariableop_66_adam_m_conv2_biasIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp%assignvariableop_67_adam_v_conv2_biasIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp$assignvariableop_68_adam_m_bn2_gammaIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp$assignvariableop_69_adam_v_bn2_gammaIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp#assignvariableop_70_adam_m_bn2_betaIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp#assignvariableop_71_adam_v_bn2_betaIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp'assignvariableop_72_adam_m_conv3_kernelIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp'assignvariableop_73_adam_v_conv3_kernelIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp%assignvariableop_74_adam_m_conv3_biasIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp%assignvariableop_75_adam_v_conv3_biasIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp$assignvariableop_76_adam_m_bn3_gammaIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp$assignvariableop_77_adam_v_bn3_gammaIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp#assignvariableop_78_adam_m_bn3_betaIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp#assignvariableop_79_adam_v_bn3_betaIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp(assignvariableop_80_adam_m_dense1_kernelIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp(assignvariableop_81_adam_v_dense1_kernelIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp&assignvariableop_82_adam_m_dense1_biasIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp&assignvariableop_83_adam_v_dense1_biasIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOpassignvariableop_84_total_1Identity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOpassignvariableop_85_count_1Identity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOpassignvariableop_86_totalIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOpassignvariableop_87_countIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_88Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_89IdentityIdentity_88:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_89Identity_89:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
_
C__inference_maxpool1_layer_call_and_return_conditional_losses_14759

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
D__inference_quant_bn2_layer_call_and_return_conditional_losses_14247

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
d
H__inference_quant_softmax_layer_call_and_return_conditional_losses_14682

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:���������
Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������
:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
d
H__inference_quant_flatten_layer_call_and_return_conditional_losses_14559

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_quant_conv2_layer_call_fn_14148

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_conv2_layer_call_and_return_conditional_losses_11880w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_quant_bn1_layer_call_and_return_conditional_losses_14061

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  :::::*
epsilon%o�:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������  �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������  : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
e
I__inference_quant_maxpool3_layer_call_and_return_conditional_losses_14543

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
J
.__inference_quant_maxpool3_layer_call_fn_14528

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_maxpool3_layer_call_and_return_conditional_losses_12059h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
J
.__inference_quant_maxpool3_layer_call_fn_14533

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_maxpool3_layer_call_and_return_conditional_losses_12370h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
)__inference_quant_bn3_layer_call_fn_14420

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_quant_bn3_layer_call_and_return_conditional_losses_12012w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_quant_relu2_layer_call_fn_14283

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_relu2_layer_call_and_return_conditional_losses_12288w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_quant_bn2_layer_call_and_return_conditional_losses_14265

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
_
C__inference_maxpool3_layer_call_and_return_conditional_losses_14903

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
��
�7
D__inference_MNIST_CNN_layer_call_and_return_conditional_losses_13759

inputsJ
@quantize_layer_allvaluesquantize_minimum_readvariableop_resource: J
@quantize_layer_allvaluesquantize_maximum_readvariableop_resource: U
;quant_conv1_lastvaluequant_batchmin_readvariableop_resource:?
1quant_conv1_lastvaluequant_assignminlast_resource:?
1quant_conv1_lastvaluequant_assignmaxlast_resource:9
+quant_conv1_biasadd_readvariableop_resource:/
!quant_bn1_readvariableop_resource:1
#quant_bn1_readvariableop_1_resource:@
2quant_bn1_fusedbatchnormv3_readvariableop_resource:B
4quant_bn1_fusedbatchnormv3_readvariableop_1_resource:L
Bquant_relu1_movingavgquantize_assignminema_readvariableop_resource: L
Bquant_relu1_movingavgquantize_assignmaxema_readvariableop_resource: U
;quant_conv2_lastvaluequant_batchmin_readvariableop_resource:?
1quant_conv2_lastvaluequant_assignminlast_resource:?
1quant_conv2_lastvaluequant_assignmaxlast_resource:9
+quant_conv2_biasadd_readvariableop_resource:/
!quant_bn2_readvariableop_resource:1
#quant_bn2_readvariableop_1_resource:@
2quant_bn2_fusedbatchnormv3_readvariableop_resource:B
4quant_bn2_fusedbatchnormv3_readvariableop_1_resource:L
Bquant_relu2_movingavgquantize_assignminema_readvariableop_resource: L
Bquant_relu2_movingavgquantize_assignmaxema_readvariableop_resource: U
;quant_conv3_lastvaluequant_batchmin_readvariableop_resource:?
1quant_conv3_lastvaluequant_assignminlast_resource:?
1quant_conv3_lastvaluequant_assignmaxlast_resource:9
+quant_conv3_biasadd_readvariableop_resource:/
!quant_bn3_readvariableop_resource:1
#quant_bn3_readvariableop_1_resource:@
2quant_bn3_fusedbatchnormv3_readvariableop_resource:B
4quant_bn3_fusedbatchnormv3_readvariableop_1_resource:L
Bquant_relu3_movingavgquantize_assignminema_readvariableop_resource: L
Bquant_relu3_movingavgquantize_assignmaxema_readvariableop_resource: O
<quant_dense1_lastvaluequant_batchmin_readvariableop_resource:	�
<
2quant_dense1_lastvaluequant_assignminlast_resource: <
2quant_dense1_lastvaluequant_assignmaxlast_resource: :
,quant_dense1_biasadd_readvariableop_resource:
M
Cquant_dense1_movingavgquantize_assignminema_readvariableop_resource: M
Cquant_dense1_movingavgquantize_assignmaxema_readvariableop_resource: 
identity��quant_bn1/AssignNewValue�quant_bn1/AssignNewValue_1�)quant_bn1/FusedBatchNormV3/ReadVariableOp�+quant_bn1/FusedBatchNormV3/ReadVariableOp_1�quant_bn1/ReadVariableOp�quant_bn1/ReadVariableOp_1�quant_bn2/AssignNewValue�quant_bn2/AssignNewValue_1�)quant_bn2/FusedBatchNormV3/ReadVariableOp�+quant_bn2/FusedBatchNormV3/ReadVariableOp_1�quant_bn2/ReadVariableOp�quant_bn2/ReadVariableOp_1�quant_bn3/AssignNewValue�quant_bn3/AssignNewValue_1�)quant_bn3/FusedBatchNormV3/ReadVariableOp�+quant_bn3/FusedBatchNormV3/ReadVariableOp_1�quant_bn3/ReadVariableOp�quant_bn3/ReadVariableOp_1�"quant_conv1/BiasAdd/ReadVariableOp�(quant_conv1/LastValueQuant/AssignMaxLast�(quant_conv1/LastValueQuant/AssignMinLast�2quant_conv1/LastValueQuant/BatchMax/ReadVariableOp�2quant_conv1/LastValueQuant/BatchMin/ReadVariableOp�Kquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp�Mquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1�Mquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2�"quant_conv2/BiasAdd/ReadVariableOp�(quant_conv2/LastValueQuant/AssignMaxLast�(quant_conv2/LastValueQuant/AssignMinLast�2quant_conv2/LastValueQuant/BatchMax/ReadVariableOp�2quant_conv2/LastValueQuant/BatchMin/ReadVariableOp�Kquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp�Mquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1�Mquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2�"quant_conv3/BiasAdd/ReadVariableOp�(quant_conv3/LastValueQuant/AssignMaxLast�(quant_conv3/LastValueQuant/AssignMinLast�2quant_conv3/LastValueQuant/BatchMax/ReadVariableOp�2quant_conv3/LastValueQuant/BatchMin/ReadVariableOp�Kquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp�Mquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1�Mquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2�#quant_dense1/BiasAdd/ReadVariableOp�)quant_dense1/LastValueQuant/AssignMaxLast�)quant_dense1/LastValueQuant/AssignMinLast�3quant_dense1/LastValueQuant/BatchMax/ReadVariableOp�3quant_dense1/LastValueQuant/BatchMin/ReadVariableOp�Bquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Dquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Dquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�?quant_dense1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�:quant_dense1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�?quant_dense1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�:quant_dense1/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Equant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Gquant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�>quant_relu1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�9quant_relu1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�>quant_relu1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�9quant_relu1/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Dquant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�>quant_relu2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�9quant_relu2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�>quant_relu2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�9quant_relu2/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Dquant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�>quant_relu3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�9quant_relu3/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�>quant_relu3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�9quant_relu3/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Dquant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�2quantize_layer/AllValuesQuantize/AssignMaxAllValue�2quantize_layer/AllValuesQuantize/AssignMinAllValue�Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp�7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp
&quantize_layer/AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
)quantize_layer/AllValuesQuantize/BatchMinMininputs/quantize_layer/AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: �
(quantize_layer/AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
)quantize_layer/AllValuesQuantize/BatchMaxMaxinputs1quantize_layer/AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: �
7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp@quantize_layer_allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype0�
(quantize_layer/AllValuesQuantize/MinimumMinimum?quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp:value:02quantize_layer/AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: q
,quantize_layer/AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
*quantize_layer/AllValuesQuantize/Minimum_1Minimum,quantize_layer/AllValuesQuantize/Minimum:z:05quantize_layer/AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: �
7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp@quantize_layer_allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype0�
(quantize_layer/AllValuesQuantize/MaximumMaximum?quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp:value:02quantize_layer/AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: q
,quantize_layer/AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
*quantize_layer/AllValuesQuantize/Maximum_1Maximum,quantize_layer/AllValuesQuantize/Maximum:z:05quantize_layer/AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: �
2quantize_layer/AllValuesQuantize/AssignMinAllValueAssignVariableOp@quantize_layer_allvaluesquantize_minimum_readvariableop_resource.quantize_layer/AllValuesQuantize/Minimum_1:z:08^quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
2quantize_layer/AllValuesQuantize/AssignMaxAllValueAssignVariableOp@quantize_layer_allvaluesquantize_maximum_readvariableop_resource.quantize_layer/AllValuesQuantize/Maximum_1:z:08^quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp@quantize_layer_allvaluesquantize_minimum_readvariableop_resource3^quantize_layer/AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype0�
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@quantize_layer_allvaluesquantize_maximum_readvariableop_resource3^quantize_layer/AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype0�
8quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsOquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:���������  �
2quant_conv1/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp;quant_conv1_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0�
5quant_conv1/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
#quant_conv1/LastValueQuant/BatchMinMin:quant_conv1/LastValueQuant/BatchMin/ReadVariableOp:value:0>quant_conv1/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:�
2quant_conv1/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp;quant_conv1_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0�
5quant_conv1/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
#quant_conv1/LastValueQuant/BatchMaxMax:quant_conv1/LastValueQuant/BatchMax/ReadVariableOp:value:0>quant_conv1/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:i
$quant_conv1/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
"quant_conv1/LastValueQuant/truedivRealDiv,quant_conv1/LastValueQuant/BatchMax:output:0-quant_conv1/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:�
"quant_conv1/LastValueQuant/MinimumMinimum,quant_conv1/LastValueQuant/BatchMin:output:0&quant_conv1/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:e
 quant_conv1/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
quant_conv1/LastValueQuant/mulMul,quant_conv1/LastValueQuant/BatchMin:output:0)quant_conv1/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:�
"quant_conv1/LastValueQuant/MaximumMaximum,quant_conv1/LastValueQuant/BatchMax:output:0"quant_conv1/LastValueQuant/mul:z:0*
T0*
_output_shapes
:�
(quant_conv1/LastValueQuant/AssignMinLastAssignVariableOp1quant_conv1_lastvaluequant_assignminlast_resource&quant_conv1/LastValueQuant/Minimum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
(quant_conv1/LastValueQuant/AssignMaxLastAssignVariableOp1quant_conv1_lastvaluequant_assignmaxlast_resource&quant_conv1/LastValueQuant/Maximum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
Kquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp;quant_conv1_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0�
Mquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp1quant_conv1_lastvaluequant_assignminlast_resource)^quant_conv1/LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0�
Mquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp1quant_conv1_lastvaluequant_assignmaxlast_resource)^quant_conv1/LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0�
<quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelSquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Uquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Uquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(�
quant_conv1/Conv2DConv2DBquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Fquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
�
"quant_conv1/BiasAdd/ReadVariableOpReadVariableOp+quant_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_conv1/BiasAddBiasAddquant_conv1/Conv2D:output:0*quant_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  v
quant_bn1/ReadVariableOpReadVariableOp!quant_bn1_readvariableop_resource*
_output_shapes
:*
dtype0z
quant_bn1/ReadVariableOp_1ReadVariableOp#quant_bn1_readvariableop_1_resource*
_output_shapes
:*
dtype0�
)quant_bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp2quant_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
+quant_bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp4quant_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
quant_bn1/FusedBatchNormV3FusedBatchNormV3quant_conv1/BiasAdd:output:0 quant_bn1/ReadVariableOp:value:0"quant_bn1/ReadVariableOp_1:value:01quant_bn1/FusedBatchNormV3/ReadVariableOp:value:03quant_bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  :::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
quant_bn1/AssignNewValueAssignVariableOp2quant_bn1_fusedbatchnormv3_readvariableop_resource'quant_bn1/FusedBatchNormV3:batch_mean:0*^quant_bn1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
quant_bn1/AssignNewValue_1AssignVariableOp4quant_bn1_fusedbatchnormv3_readvariableop_1_resource+quant_bn1/FusedBatchNormV3:batch_variance:0,^quant_bn1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(r
quant_relu1/ReluReluquant_bn1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������  |
#quant_relu1/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
&quant_relu1/MovingAvgQuantize/BatchMinMinquant_relu1/Relu:activations:0,quant_relu1/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: ~
%quant_relu1/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
&quant_relu1/MovingAvgQuantize/BatchMaxMaxquant_relu1/Relu:activations:0.quant_relu1/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: l
'quant_relu1/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
%quant_relu1/MovingAvgQuantize/MinimumMinimum/quant_relu1/MovingAvgQuantize/BatchMin:output:00quant_relu1/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: l
'quant_relu1/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
%quant_relu1/MovingAvgQuantize/MaximumMaximum/quant_relu1/MovingAvgQuantize/BatchMax:output:00quant_relu1/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: u
0quant_relu1/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
9quant_relu1/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpBquant_relu1_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
.quant_relu1/MovingAvgQuantize/AssignMinEma/subSubAquant_relu1/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0)quant_relu1/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
.quant_relu1/MovingAvgQuantize/AssignMinEma/mulMul2quant_relu1/MovingAvgQuantize/AssignMinEma/sub:z:09quant_relu1/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
>quant_relu1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpBquant_relu1_movingavgquantize_assignminema_readvariableop_resource2quant_relu1/MovingAvgQuantize/AssignMinEma/mul:z:0:^quant_relu1/MovingAvgQuantize/AssignMinEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0u
0quant_relu1/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
9quant_relu1/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpBquant_relu1_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
.quant_relu1/MovingAvgQuantize/AssignMaxEma/subSubAquant_relu1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0)quant_relu1/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
.quant_relu1/MovingAvgQuantize/AssignMaxEma/mulMul2quant_relu1/MovingAvgQuantize/AssignMaxEma/sub:z:09quant_relu1/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
>quant_relu1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpBquant_relu1_movingavgquantize_assignmaxema_readvariableop_resource2quant_relu1/MovingAvgQuantize/AssignMaxEma/mul:z:0:^quant_relu1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
Dquant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpBquant_relu1_movingavgquantize_assignminema_readvariableop_resource?^quant_relu1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Fquant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpBquant_relu1_movingavgquantize_assignmaxema_readvariableop_resource?^quant_relu1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
5quant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_relu1/Relu:activations:0Lquant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:���������  �
quant_maxpool1/MaxPoolMaxPool?quant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
�
2quant_conv2/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp;quant_conv2_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0�
5quant_conv2/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
#quant_conv2/LastValueQuant/BatchMinMin:quant_conv2/LastValueQuant/BatchMin/ReadVariableOp:value:0>quant_conv2/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:�
2quant_conv2/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp;quant_conv2_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0�
5quant_conv2/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
#quant_conv2/LastValueQuant/BatchMaxMax:quant_conv2/LastValueQuant/BatchMax/ReadVariableOp:value:0>quant_conv2/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:i
$quant_conv2/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
"quant_conv2/LastValueQuant/truedivRealDiv,quant_conv2/LastValueQuant/BatchMax:output:0-quant_conv2/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:�
"quant_conv2/LastValueQuant/MinimumMinimum,quant_conv2/LastValueQuant/BatchMin:output:0&quant_conv2/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:e
 quant_conv2/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
quant_conv2/LastValueQuant/mulMul,quant_conv2/LastValueQuant/BatchMin:output:0)quant_conv2/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:�
"quant_conv2/LastValueQuant/MaximumMaximum,quant_conv2/LastValueQuant/BatchMax:output:0"quant_conv2/LastValueQuant/mul:z:0*
T0*
_output_shapes
:�
(quant_conv2/LastValueQuant/AssignMinLastAssignVariableOp1quant_conv2_lastvaluequant_assignminlast_resource&quant_conv2/LastValueQuant/Minimum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
(quant_conv2/LastValueQuant/AssignMaxLastAssignVariableOp1quant_conv2_lastvaluequant_assignmaxlast_resource&quant_conv2/LastValueQuant/Maximum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
Kquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp;quant_conv2_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0�
Mquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp1quant_conv2_lastvaluequant_assignminlast_resource)^quant_conv2/LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0�
Mquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp1quant_conv2_lastvaluequant_assignmaxlast_resource)^quant_conv2/LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0�
<quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelSquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Uquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Uquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(�
quant_conv2/Conv2DConv2Dquant_maxpool1/MaxPool:output:0Fquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
"quant_conv2/BiasAdd/ReadVariableOpReadVariableOp+quant_conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_conv2/BiasAddBiasAddquant_conv2/Conv2D:output:0*quant_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v
quant_bn2/ReadVariableOpReadVariableOp!quant_bn2_readvariableop_resource*
_output_shapes
:*
dtype0z
quant_bn2/ReadVariableOp_1ReadVariableOp#quant_bn2_readvariableop_1_resource*
_output_shapes
:*
dtype0�
)quant_bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp2quant_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
+quant_bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp4quant_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
quant_bn2/FusedBatchNormV3FusedBatchNormV3quant_conv2/BiasAdd:output:0 quant_bn2/ReadVariableOp:value:0"quant_bn2/ReadVariableOp_1:value:01quant_bn2/FusedBatchNormV3/ReadVariableOp:value:03quant_bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
quant_bn2/AssignNewValueAssignVariableOp2quant_bn2_fusedbatchnormv3_readvariableop_resource'quant_bn2/FusedBatchNormV3:batch_mean:0*^quant_bn2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
quant_bn2/AssignNewValue_1AssignVariableOp4quant_bn2_fusedbatchnormv3_readvariableop_1_resource+quant_bn2/FusedBatchNormV3:batch_variance:0,^quant_bn2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(r
quant_relu2/ReluReluquant_bn2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������|
#quant_relu2/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
&quant_relu2/MovingAvgQuantize/BatchMinMinquant_relu2/Relu:activations:0,quant_relu2/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: ~
%quant_relu2/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
&quant_relu2/MovingAvgQuantize/BatchMaxMaxquant_relu2/Relu:activations:0.quant_relu2/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: l
'quant_relu2/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
%quant_relu2/MovingAvgQuantize/MinimumMinimum/quant_relu2/MovingAvgQuantize/BatchMin:output:00quant_relu2/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: l
'quant_relu2/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
%quant_relu2/MovingAvgQuantize/MaximumMaximum/quant_relu2/MovingAvgQuantize/BatchMax:output:00quant_relu2/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: u
0quant_relu2/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
9quant_relu2/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpBquant_relu2_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
.quant_relu2/MovingAvgQuantize/AssignMinEma/subSubAquant_relu2/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0)quant_relu2/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
.quant_relu2/MovingAvgQuantize/AssignMinEma/mulMul2quant_relu2/MovingAvgQuantize/AssignMinEma/sub:z:09quant_relu2/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
>quant_relu2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpBquant_relu2_movingavgquantize_assignminema_readvariableop_resource2quant_relu2/MovingAvgQuantize/AssignMinEma/mul:z:0:^quant_relu2/MovingAvgQuantize/AssignMinEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0u
0quant_relu2/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
9quant_relu2/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpBquant_relu2_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
.quant_relu2/MovingAvgQuantize/AssignMaxEma/subSubAquant_relu2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0)quant_relu2/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
.quant_relu2/MovingAvgQuantize/AssignMaxEma/mulMul2quant_relu2/MovingAvgQuantize/AssignMaxEma/sub:z:09quant_relu2/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
>quant_relu2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpBquant_relu2_movingavgquantize_assignmaxema_readvariableop_resource2quant_relu2/MovingAvgQuantize/AssignMaxEma/mul:z:0:^quant_relu2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
Dquant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpBquant_relu2_movingavgquantize_assignminema_readvariableop_resource?^quant_relu2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Fquant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpBquant_relu2_movingavgquantize_assignmaxema_readvariableop_resource?^quant_relu2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
5quant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_relu2/Relu:activations:0Lquant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:����������
quant_maxpool2/MaxPoolMaxPool?quant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
�
2quant_conv3/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp;quant_conv3_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0�
5quant_conv3/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
#quant_conv3/LastValueQuant/BatchMinMin:quant_conv3/LastValueQuant/BatchMin/ReadVariableOp:value:0>quant_conv3/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:�
2quant_conv3/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp;quant_conv3_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0�
5quant_conv3/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
#quant_conv3/LastValueQuant/BatchMaxMax:quant_conv3/LastValueQuant/BatchMax/ReadVariableOp:value:0>quant_conv3/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:i
$quant_conv3/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
"quant_conv3/LastValueQuant/truedivRealDiv,quant_conv3/LastValueQuant/BatchMax:output:0-quant_conv3/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:�
"quant_conv3/LastValueQuant/MinimumMinimum,quant_conv3/LastValueQuant/BatchMin:output:0&quant_conv3/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:e
 quant_conv3/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
quant_conv3/LastValueQuant/mulMul,quant_conv3/LastValueQuant/BatchMin:output:0)quant_conv3/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:�
"quant_conv3/LastValueQuant/MaximumMaximum,quant_conv3/LastValueQuant/BatchMax:output:0"quant_conv3/LastValueQuant/mul:z:0*
T0*
_output_shapes
:�
(quant_conv3/LastValueQuant/AssignMinLastAssignVariableOp1quant_conv3_lastvaluequant_assignminlast_resource&quant_conv3/LastValueQuant/Minimum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
(quant_conv3/LastValueQuant/AssignMaxLastAssignVariableOp1quant_conv3_lastvaluequant_assignmaxlast_resource&quant_conv3/LastValueQuant/Maximum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
Kquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp;quant_conv3_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0�
Mquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp1quant_conv3_lastvaluequant_assignminlast_resource)^quant_conv3/LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0�
Mquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp1quant_conv3_lastvaluequant_assignmaxlast_resource)^quant_conv3/LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0�
<quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelSquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Uquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Uquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(�
quant_conv3/Conv2DConv2Dquant_maxpool2/MaxPool:output:0Fquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
"quant_conv3/BiasAdd/ReadVariableOpReadVariableOp+quant_conv3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_conv3/BiasAddBiasAddquant_conv3/Conv2D:output:0*quant_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v
quant_bn3/ReadVariableOpReadVariableOp!quant_bn3_readvariableop_resource*
_output_shapes
:*
dtype0z
quant_bn3/ReadVariableOp_1ReadVariableOp#quant_bn3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
)quant_bn3/FusedBatchNormV3/ReadVariableOpReadVariableOp2quant_bn3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
+quant_bn3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp4quant_bn3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
quant_bn3/FusedBatchNormV3FusedBatchNormV3quant_conv3/BiasAdd:output:0 quant_bn3/ReadVariableOp:value:0"quant_bn3/ReadVariableOp_1:value:01quant_bn3/FusedBatchNormV3/ReadVariableOp:value:03quant_bn3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
quant_bn3/AssignNewValueAssignVariableOp2quant_bn3_fusedbatchnormv3_readvariableop_resource'quant_bn3/FusedBatchNormV3:batch_mean:0*^quant_bn3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
quant_bn3/AssignNewValue_1AssignVariableOp4quant_bn3_fusedbatchnormv3_readvariableop_1_resource+quant_bn3/FusedBatchNormV3:batch_variance:0,^quant_bn3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(r
quant_relu3/ReluReluquant_bn3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������|
#quant_relu3/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
&quant_relu3/MovingAvgQuantize/BatchMinMinquant_relu3/Relu:activations:0,quant_relu3/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: ~
%quant_relu3/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
&quant_relu3/MovingAvgQuantize/BatchMaxMaxquant_relu3/Relu:activations:0.quant_relu3/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: l
'quant_relu3/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
%quant_relu3/MovingAvgQuantize/MinimumMinimum/quant_relu3/MovingAvgQuantize/BatchMin:output:00quant_relu3/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: l
'quant_relu3/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
%quant_relu3/MovingAvgQuantize/MaximumMaximum/quant_relu3/MovingAvgQuantize/BatchMax:output:00quant_relu3/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: u
0quant_relu3/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
9quant_relu3/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpBquant_relu3_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
.quant_relu3/MovingAvgQuantize/AssignMinEma/subSubAquant_relu3/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0)quant_relu3/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
.quant_relu3/MovingAvgQuantize/AssignMinEma/mulMul2quant_relu3/MovingAvgQuantize/AssignMinEma/sub:z:09quant_relu3/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
>quant_relu3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpBquant_relu3_movingavgquantize_assignminema_readvariableop_resource2quant_relu3/MovingAvgQuantize/AssignMinEma/mul:z:0:^quant_relu3/MovingAvgQuantize/AssignMinEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0u
0quant_relu3/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
9quant_relu3/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpBquant_relu3_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
.quant_relu3/MovingAvgQuantize/AssignMaxEma/subSubAquant_relu3/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0)quant_relu3/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
.quant_relu3/MovingAvgQuantize/AssignMaxEma/mulMul2quant_relu3/MovingAvgQuantize/AssignMaxEma/sub:z:09quant_relu3/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
>quant_relu3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpBquant_relu3_movingavgquantize_assignmaxema_readvariableop_resource2quant_relu3/MovingAvgQuantize/AssignMaxEma/mul:z:0:^quant_relu3/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
Dquant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpBquant_relu3_movingavgquantize_assignminema_readvariableop_resource?^quant_relu3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Fquant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpBquant_relu3_movingavgquantize_assignmaxema_readvariableop_resource?^quant_relu3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
5quant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_relu3/Relu:activations:0Lquant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:����������
quant_maxpool3/MaxPoolMaxPool?quant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
d
quant_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
quant_flatten/ReshapeReshapequant_maxpool3/MaxPool:output:0quant_flatten/Const:output:0*
T0*(
_output_shapes
:����������r
!quant_dense1/LastValueQuant/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
3quant_dense1/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp<quant_dense1_lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
$quant_dense1/LastValueQuant/BatchMinMin;quant_dense1/LastValueQuant/BatchMin/ReadVariableOp:value:0*quant_dense1/LastValueQuant/Const:output:0*
T0*
_output_shapes
: t
#quant_dense1/LastValueQuant/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
3quant_dense1/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp<quant_dense1_lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
$quant_dense1/LastValueQuant/BatchMaxMax;quant_dense1/LastValueQuant/BatchMax/ReadVariableOp:value:0,quant_dense1/LastValueQuant/Const_1:output:0*
T0*
_output_shapes
: j
%quant_dense1/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
#quant_dense1/LastValueQuant/truedivRealDiv-quant_dense1/LastValueQuant/BatchMax:output:0.quant_dense1/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
#quant_dense1/LastValueQuant/MinimumMinimum-quant_dense1/LastValueQuant/BatchMin:output:0'quant_dense1/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: f
!quant_dense1/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
quant_dense1/LastValueQuant/mulMul-quant_dense1/LastValueQuant/BatchMin:output:0*quant_dense1/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: �
#quant_dense1/LastValueQuant/MaximumMaximum-quant_dense1/LastValueQuant/BatchMax:output:0#quant_dense1/LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
)quant_dense1/LastValueQuant/AssignMinLastAssignVariableOp2quant_dense1_lastvaluequant_assignminlast_resource'quant_dense1/LastValueQuant/Minimum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
)quant_dense1/LastValueQuant/AssignMaxLastAssignVariableOp2quant_dense1_lastvaluequant_assignmaxlast_resource'quant_dense1/LastValueQuant/Maximum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
Bquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp<quant_dense1_lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
Dquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp2quant_dense1_lastvaluequant_assignminlast_resource*^quant_dense1/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
Dquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp2quant_dense1_lastvaluequant_assignmaxlast_resource*^quant_dense1/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
3quant_dense1/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsJquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Lquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Lquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	�
*
narrow_range(�
quant_dense1/MatMulMatMulquant_flatten/Reshape:output:0=quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������
�
#quant_dense1/BiasAdd/ReadVariableOpReadVariableOp,quant_dense1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
quant_dense1/BiasAddBiasAddquant_dense1/MatMul:product:0+quant_dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
u
$quant_dense1/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
'quant_dense1/MovingAvgQuantize/BatchMinMinquant_dense1/BiasAdd:output:0-quant_dense1/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: w
&quant_dense1/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
'quant_dense1/MovingAvgQuantize/BatchMaxMaxquant_dense1/BiasAdd:output:0/quant_dense1/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: m
(quant_dense1/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&quant_dense1/MovingAvgQuantize/MinimumMinimum0quant_dense1/MovingAvgQuantize/BatchMin:output:01quant_dense1/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: m
(quant_dense1/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&quant_dense1/MovingAvgQuantize/MaximumMaximum0quant_dense1/MovingAvgQuantize/BatchMax:output:01quant_dense1/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: v
1quant_dense1/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
:quant_dense1/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpCquant_dense1_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
/quant_dense1/MovingAvgQuantize/AssignMinEma/subSubBquant_dense1/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0*quant_dense1/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
/quant_dense1/MovingAvgQuantize/AssignMinEma/mulMul3quant_dense1/MovingAvgQuantize/AssignMinEma/sub:z:0:quant_dense1/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
?quant_dense1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpCquant_dense1_movingavgquantize_assignminema_readvariableop_resource3quant_dense1/MovingAvgQuantize/AssignMinEma/mul:z:0;^quant_dense1/MovingAvgQuantize/AssignMinEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0v
1quant_dense1/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
:quant_dense1/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpCquant_dense1_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
/quant_dense1/MovingAvgQuantize/AssignMaxEma/subSubBquant_dense1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0*quant_dense1/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
/quant_dense1/MovingAvgQuantize/AssignMaxEma/mulMul3quant_dense1/MovingAvgQuantize/AssignMaxEma/sub:z:0:quant_dense1/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
?quant_dense1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpCquant_dense1_movingavgquantize_assignmaxema_readvariableop_resource3quant_dense1/MovingAvgQuantize/AssignMaxEma/mul:z:0;^quant_dense1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
Equant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpCquant_dense1_movingavgquantize_assignminema_readvariableop_resource@^quant_dense1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Gquant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCquant_dense1_movingavgquantize_assignmaxema_readvariableop_resource@^quant_dense1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
6quant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_dense1/BiasAdd:output:0Mquant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Oquant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������
�
quant_softmax/SoftmaxSoftmax@quant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������
n
IdentityIdentityquant_softmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�"
NoOpNoOp^quant_bn1/AssignNewValue^quant_bn1/AssignNewValue_1*^quant_bn1/FusedBatchNormV3/ReadVariableOp,^quant_bn1/FusedBatchNormV3/ReadVariableOp_1^quant_bn1/ReadVariableOp^quant_bn1/ReadVariableOp_1^quant_bn2/AssignNewValue^quant_bn2/AssignNewValue_1*^quant_bn2/FusedBatchNormV3/ReadVariableOp,^quant_bn2/FusedBatchNormV3/ReadVariableOp_1^quant_bn2/ReadVariableOp^quant_bn2/ReadVariableOp_1^quant_bn3/AssignNewValue^quant_bn3/AssignNewValue_1*^quant_bn3/FusedBatchNormV3/ReadVariableOp,^quant_bn3/FusedBatchNormV3/ReadVariableOp_1^quant_bn3/ReadVariableOp^quant_bn3/ReadVariableOp_1#^quant_conv1/BiasAdd/ReadVariableOp)^quant_conv1/LastValueQuant/AssignMaxLast)^quant_conv1/LastValueQuant/AssignMinLast3^quant_conv1/LastValueQuant/BatchMax/ReadVariableOp3^quant_conv1/LastValueQuant/BatchMin/ReadVariableOpL^quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpN^quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1N^quant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2#^quant_conv2/BiasAdd/ReadVariableOp)^quant_conv2/LastValueQuant/AssignMaxLast)^quant_conv2/LastValueQuant/AssignMinLast3^quant_conv2/LastValueQuant/BatchMax/ReadVariableOp3^quant_conv2/LastValueQuant/BatchMin/ReadVariableOpL^quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpN^quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1N^quant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2#^quant_conv3/BiasAdd/ReadVariableOp)^quant_conv3/LastValueQuant/AssignMaxLast)^quant_conv3/LastValueQuant/AssignMinLast3^quant_conv3/LastValueQuant/BatchMax/ReadVariableOp3^quant_conv3/LastValueQuant/BatchMin/ReadVariableOpL^quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpN^quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1N^quant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2$^quant_dense1/BiasAdd/ReadVariableOp*^quant_dense1/LastValueQuant/AssignMaxLast*^quant_dense1/LastValueQuant/AssignMinLast4^quant_dense1/LastValueQuant/BatchMax/ReadVariableOp4^quant_dense1/LastValueQuant/BatchMin/ReadVariableOpC^quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpE^quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1E^quant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2@^quant_dense1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp;^quant_dense1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp@^quant_dense1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp;^quant_dense1/MovingAvgQuantize/AssignMinEma/ReadVariableOpF^quant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpH^quant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?^quant_relu1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp:^quant_relu1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp?^quant_relu1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp:^quant_relu1/MovingAvgQuantize/AssignMinEma/ReadVariableOpE^quant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?^quant_relu2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp:^quant_relu2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp?^quant_relu2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp:^quant_relu2/MovingAvgQuantize/AssignMinEma/ReadVariableOpE^quant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?^quant_relu3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp:^quant_relu3/MovingAvgQuantize/AssignMaxEma/ReadVariableOp?^quant_relu3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp:^quant_relu3/MovingAvgQuantize/AssignMinEma/ReadVariableOpE^quant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_13^quantize_layer/AllValuesQuantize/AssignMaxAllValue3^quantize_layer/AllValuesQuantize/AssignMinAllValueH^quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_18^quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp8^quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 24
quant_bn1/AssignNewValuequant_bn1/AssignNewValue28
quant_bn1/AssignNewValue_1quant_bn1/AssignNewValue_12V
)quant_bn1/FusedBatchNormV3/ReadVariableOp)quant_bn1/FusedBatchNormV3/ReadVariableOp2Z
+quant_bn1/FusedBatchNormV3/ReadVariableOp_1+quant_bn1/FusedBatchNormV3/ReadVariableOp_124
quant_bn1/ReadVariableOpquant_bn1/ReadVariableOp28
quant_bn1/ReadVariableOp_1quant_bn1/ReadVariableOp_124
quant_bn2/AssignNewValuequant_bn2/AssignNewValue28
quant_bn2/AssignNewValue_1quant_bn2/AssignNewValue_12V
)quant_bn2/FusedBatchNormV3/ReadVariableOp)quant_bn2/FusedBatchNormV3/ReadVariableOp2Z
+quant_bn2/FusedBatchNormV3/ReadVariableOp_1+quant_bn2/FusedBatchNormV3/ReadVariableOp_124
quant_bn2/ReadVariableOpquant_bn2/ReadVariableOp28
quant_bn2/ReadVariableOp_1quant_bn2/ReadVariableOp_124
quant_bn3/AssignNewValuequant_bn3/AssignNewValue28
quant_bn3/AssignNewValue_1quant_bn3/AssignNewValue_12V
)quant_bn3/FusedBatchNormV3/ReadVariableOp)quant_bn3/FusedBatchNormV3/ReadVariableOp2Z
+quant_bn3/FusedBatchNormV3/ReadVariableOp_1+quant_bn3/FusedBatchNormV3/ReadVariableOp_124
quant_bn3/ReadVariableOpquant_bn3/ReadVariableOp28
quant_bn3/ReadVariableOp_1quant_bn3/ReadVariableOp_12H
"quant_conv1/BiasAdd/ReadVariableOp"quant_conv1/BiasAdd/ReadVariableOp2T
(quant_conv1/LastValueQuant/AssignMaxLast(quant_conv1/LastValueQuant/AssignMaxLast2T
(quant_conv1/LastValueQuant/AssignMinLast(quant_conv1/LastValueQuant/AssignMinLast2h
2quant_conv1/LastValueQuant/BatchMax/ReadVariableOp2quant_conv1/LastValueQuant/BatchMax/ReadVariableOp2h
2quant_conv1/LastValueQuant/BatchMin/ReadVariableOp2quant_conv1/LastValueQuant/BatchMin/ReadVariableOp2�
Kquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpKquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2�
Mquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Mquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12�
Mquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Mquant_conv1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22H
"quant_conv2/BiasAdd/ReadVariableOp"quant_conv2/BiasAdd/ReadVariableOp2T
(quant_conv2/LastValueQuant/AssignMaxLast(quant_conv2/LastValueQuant/AssignMaxLast2T
(quant_conv2/LastValueQuant/AssignMinLast(quant_conv2/LastValueQuant/AssignMinLast2h
2quant_conv2/LastValueQuant/BatchMax/ReadVariableOp2quant_conv2/LastValueQuant/BatchMax/ReadVariableOp2h
2quant_conv2/LastValueQuant/BatchMin/ReadVariableOp2quant_conv2/LastValueQuant/BatchMin/ReadVariableOp2�
Kquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpKquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2�
Mquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Mquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12�
Mquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Mquant_conv2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22H
"quant_conv3/BiasAdd/ReadVariableOp"quant_conv3/BiasAdd/ReadVariableOp2T
(quant_conv3/LastValueQuant/AssignMaxLast(quant_conv3/LastValueQuant/AssignMaxLast2T
(quant_conv3/LastValueQuant/AssignMinLast(quant_conv3/LastValueQuant/AssignMinLast2h
2quant_conv3/LastValueQuant/BatchMax/ReadVariableOp2quant_conv3/LastValueQuant/BatchMax/ReadVariableOp2h
2quant_conv3/LastValueQuant/BatchMin/ReadVariableOp2quant_conv3/LastValueQuant/BatchMin/ReadVariableOp2�
Kquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpKquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2�
Mquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Mquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12�
Mquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Mquant_conv3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22J
#quant_dense1/BiasAdd/ReadVariableOp#quant_dense1/BiasAdd/ReadVariableOp2V
)quant_dense1/LastValueQuant/AssignMaxLast)quant_dense1/LastValueQuant/AssignMaxLast2V
)quant_dense1/LastValueQuant/AssignMinLast)quant_dense1/LastValueQuant/AssignMinLast2j
3quant_dense1/LastValueQuant/BatchMax/ReadVariableOp3quant_dense1/LastValueQuant/BatchMax/ReadVariableOp2j
3quant_dense1/LastValueQuant/BatchMin/ReadVariableOp3quant_dense1/LastValueQuant/BatchMin/ReadVariableOp2�
Bquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpBquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Dquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Dquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Dquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Dquant_dense1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
?quant_dense1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?quant_dense1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2x
:quant_dense1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:quant_dense1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
?quant_dense1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?quant_dense1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2x
:quant_dense1/MovingAvgQuantize/AssignMinEma/ReadVariableOp:quant_dense1/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Equant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpEquant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Gquant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Gquant_dense1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
>quant_relu1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp>quant_relu1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2v
9quant_relu1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp9quant_relu1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
>quant_relu1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp>quant_relu1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2v
9quant_relu1/MovingAvgQuantize/AssignMinEma/ReadVariableOp9quant_relu1/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Dquant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpDquant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_relu1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
>quant_relu2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp>quant_relu2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2v
9quant_relu2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp9quant_relu2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
>quant_relu2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp>quant_relu2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2v
9quant_relu2/MovingAvgQuantize/AssignMinEma/ReadVariableOp9quant_relu2/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Dquant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpDquant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_relu2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
>quant_relu3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp>quant_relu3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2v
9quant_relu3/MovingAvgQuantize/AssignMaxEma/ReadVariableOp9quant_relu3/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
>quant_relu3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp>quant_relu3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2v
9quant_relu3/MovingAvgQuantize/AssignMinEma/ReadVariableOp9quant_relu3/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Dquant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpDquant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_relu3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12h
2quantize_layer/AllValuesQuantize/AssignMaxAllValue2quantize_layer/AllValuesQuantize/AssignMaxAllValue2h
2quantize_layer/AllValuesQuantize/AssignMinAllValue2quantize_layer/AllValuesQuantize/AssignMinAllValue2�
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp2r
7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
>__inference_bn2_layer_call_and_return_conditional_losses_14803

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�*
�
F__inference_quant_relu3_layer_call_and_return_conditional_losses_12048

inputs@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1N
ReluReluinputs*
T0*/
_output_shapes
:���������p
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
G__inference_quant_dense1_layer_call_and_return_conditional_losses_14667

inputsQ
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:	�
J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:
K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	�
*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������
�
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�#
�
I__inference_quantize_layer_layer_call_and_return_conditional_losses_13922

inputs;
1allvaluesquantize_minimum_readvariableop_resource: ;
1allvaluesquantize_maximum_readvariableop_resource: 
identity��#AllValuesQuantize/AssignMaxAllValue�#AllValuesQuantize/AssignMinAllValue�8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�(AllValuesQuantize/Maximum/ReadVariableOp�(AllValuesQuantize/Minimum/ReadVariableOpp
AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             l
AllValuesQuantize/BatchMinMininputs AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: r
AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             n
AllValuesQuantize/BatchMaxMaxinputs"AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: �
(AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype0�
AllValuesQuantize/MinimumMinimum0AllValuesQuantize/Minimum/ReadVariableOp:value:0#AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
AllValuesQuantize/Minimum_1MinimumAllValuesQuantize/Minimum:z:0&AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: �
(AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp1allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype0�
AllValuesQuantize/MaximumMaximum0AllValuesQuantize/Maximum/ReadVariableOp:value:0#AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
AllValuesQuantize/Maximum_1MaximumAllValuesQuantize/Maximum:z:0&AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: �
#AllValuesQuantize/AssignMinAllValueAssignVariableOp1allvaluesquantize_minimum_readvariableop_resourceAllValuesQuantize/Minimum_1:z:0)^AllValuesQuantize/Minimum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
#AllValuesQuantize/AssignMaxAllValueAssignVariableOp1allvaluesquantize_maximum_readvariableop_resourceAllValuesQuantize/Maximum_1:z:0)^AllValuesQuantize/Maximum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource$^AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype0�
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp1allvaluesquantize_maximum_readvariableop_resource$^AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype0�
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:���������  �
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:���������  �
NoOpNoOp$^AllValuesQuantize/AssignMaxAllValue$^AllValuesQuantize/AssignMinAllValue9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1)^AllValuesQuantize/Maximum/ReadVariableOp)^AllValuesQuantize/Minimum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 2J
#AllValuesQuantize/AssignMaxAllValue#AllValuesQuantize/AssignMaxAllValue2J
#AllValuesQuantize/AssignMinAllValue#AllValuesQuantize/AssignMinAllValue2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12T
(AllValuesQuantize/Maximum/ReadVariableOp(AllValuesQuantize/Maximum/ReadVariableOp2T
(AllValuesQuantize/Minimum/ReadVariableOp(AllValuesQuantize/Minimum/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
>__inference_bn1_layer_call_and_return_conditional_losses_11527

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
)__inference_quant_bn3_layer_call_fn_14433

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_quant_bn3_layer_call_and_return_conditional_losses_12341w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
>__inference_bn2_layer_call_and_return_conditional_losses_11603

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
F__inference_quant_relu2_layer_call_and_return_conditional_losses_14319

inputsK
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1N
ReluReluinputs*
T0*/
_output_shapes
:����������
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_quant_relu2_layer_call_and_return_conditional_losses_12288

inputsK
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1N
ReluReluinputs*
T0*/
_output_shapes
:����������
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_quantize_layer_layer_call_fn_13892

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quantize_layer_layer_call_and_return_conditional_losses_11743w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�O
�
D__inference_MNIST_CNN_layer_call_and_return_conditional_losses_12139	
input
quantize_layer_11744: 
quantize_layer_11746: +
quant_conv1_11777:
quant_conv1_11779:
quant_conv1_11781:
quant_conv1_11783:
quant_bn1_11805:
quant_bn1_11807:
quant_bn1_11809:
quant_bn1_11811:
quant_relu1_11841: 
quant_relu1_11843: +
quant_conv2_11881:
quant_conv2_11883:
quant_conv2_11885:
quant_conv2_11887:
quant_bn2_11909:
quant_bn2_11911:
quant_bn2_11913:
quant_bn2_11915:
quant_relu2_11945: 
quant_relu2_11947: +
quant_conv3_11985:
quant_conv3_11987:
quant_conv3_11989:
quant_conv3_11991:
quant_bn3_12013:
quant_bn3_12015:
quant_bn3_12017:
quant_bn3_12019:
quant_relu3_12049: 
quant_relu3_12051: %
quant_dense1_12118:	�

quant_dense1_12120: 
quant_dense1_12122:  
quant_dense1_12124:

quant_dense1_12126: 
quant_dense1_12128: 
identity��!quant_bn1/StatefulPartitionedCall�!quant_bn2/StatefulPartitionedCall�!quant_bn3/StatefulPartitionedCall�#quant_conv1/StatefulPartitionedCall�#quant_conv2/StatefulPartitionedCall�#quant_conv3/StatefulPartitionedCall�$quant_dense1/StatefulPartitionedCall�#quant_relu1/StatefulPartitionedCall�#quant_relu2/StatefulPartitionedCall�#quant_relu3/StatefulPartitionedCall�&quantize_layer/StatefulPartitionedCall�
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallinputquantize_layer_11744quantize_layer_11746*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quantize_layer_layer_call_and_return_conditional_losses_11743�
#quant_conv1/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv1_11777quant_conv1_11779quant_conv1_11781quant_conv1_11783*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_conv1_layer_call_and_return_conditional_losses_11776�
!quant_bn1/StatefulPartitionedCallStatefulPartitionedCall,quant_conv1/StatefulPartitionedCall:output:0quant_bn1_11805quant_bn1_11807quant_bn1_11809quant_bn1_11811*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_quant_bn1_layer_call_and_return_conditional_losses_11804�
#quant_relu1/StatefulPartitionedCallStatefulPartitionedCall*quant_bn1/StatefulPartitionedCall:output:0quant_relu1_11841quant_relu1_11843*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_relu1_layer_call_and_return_conditional_losses_11840�
quant_maxpool1/PartitionedCallPartitionedCall,quant_relu1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_maxpool1_layer_call_and_return_conditional_losses_11851�
#quant_conv2/StatefulPartitionedCallStatefulPartitionedCall'quant_maxpool1/PartitionedCall:output:0quant_conv2_11881quant_conv2_11883quant_conv2_11885quant_conv2_11887*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_conv2_layer_call_and_return_conditional_losses_11880�
!quant_bn2/StatefulPartitionedCallStatefulPartitionedCall,quant_conv2/StatefulPartitionedCall:output:0quant_bn2_11909quant_bn2_11911quant_bn2_11913quant_bn2_11915*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_quant_bn2_layer_call_and_return_conditional_losses_11908�
#quant_relu2/StatefulPartitionedCallStatefulPartitionedCall*quant_bn2/StatefulPartitionedCall:output:0quant_relu2_11945quant_relu2_11947*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_relu2_layer_call_and_return_conditional_losses_11944�
quant_maxpool2/PartitionedCallPartitionedCall,quant_relu2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_maxpool2_layer_call_and_return_conditional_losses_11955�
#quant_conv3/StatefulPartitionedCallStatefulPartitionedCall'quant_maxpool2/PartitionedCall:output:0quant_conv3_11985quant_conv3_11987quant_conv3_11989quant_conv3_11991*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_conv3_layer_call_and_return_conditional_losses_11984�
!quant_bn3/StatefulPartitionedCallStatefulPartitionedCall,quant_conv3/StatefulPartitionedCall:output:0quant_bn3_12013quant_bn3_12015quant_bn3_12017quant_bn3_12019*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_quant_bn3_layer_call_and_return_conditional_losses_12012�
#quant_relu3/StatefulPartitionedCallStatefulPartitionedCall*quant_bn3/StatefulPartitionedCall:output:0quant_relu3_12049quant_relu3_12051*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_relu3_layer_call_and_return_conditional_losses_12048�
quant_maxpool3/PartitionedCallPartitionedCall,quant_relu3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_maxpool3_layer_call_and_return_conditional_losses_12059�
quant_flatten/PartitionedCallPartitionedCall'quant_maxpool3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_quant_flatten_layer_call_and_return_conditional_losses_12067�
$quant_dense1/StatefulPartitionedCallStatefulPartitionedCall&quant_flatten/PartitionedCall:output:0quant_dense1_12118quant_dense1_12120quant_dense1_12122quant_dense1_12124quant_dense1_12126quant_dense1_12128*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_quant_dense1_layer_call_and_return_conditional_losses_12117�
quant_softmax/PartitionedCallPartitionedCall-quant_dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_quant_softmax_layer_call_and_return_conditional_losses_12136u
IdentityIdentity&quant_softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp"^quant_bn1/StatefulPartitionedCall"^quant_bn2/StatefulPartitionedCall"^quant_bn3/StatefulPartitionedCall$^quant_conv1/StatefulPartitionedCall$^quant_conv2/StatefulPartitionedCall$^quant_conv3/StatefulPartitionedCall%^quant_dense1/StatefulPartitionedCall$^quant_relu1/StatefulPartitionedCall$^quant_relu2/StatefulPartitionedCall$^quant_relu3/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!quant_bn1/StatefulPartitionedCall!quant_bn1/StatefulPartitionedCall2F
!quant_bn2/StatefulPartitionedCall!quant_bn2/StatefulPartitionedCall2F
!quant_bn3/StatefulPartitionedCall!quant_bn3/StatefulPartitionedCall2J
#quant_conv1/StatefulPartitionedCall#quant_conv1/StatefulPartitionedCall2J
#quant_conv2/StatefulPartitionedCall#quant_conv2/StatefulPartitionedCall2J
#quant_conv3/StatefulPartitionedCall#quant_conv3/StatefulPartitionedCall2L
$quant_dense1/StatefulPartitionedCall$quant_dense1/StatefulPartitionedCall2J
#quant_relu1/StatefulPartitionedCall#quant_relu1/StatefulPartitionedCall2J
#quant_relu2/StatefulPartitionedCall#quant_relu2/StatefulPartitionedCall2J
#quant_relu3/StatefulPartitionedCall#quant_relu3/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:V R
/
_output_shapes
:���������  

_user_specified_nameinput
�
_
C__inference_maxpool3_layer_call_and_return_conditional_losses_11712

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
+__inference_quant_relu3_layer_call_fn_14478

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_relu3_layer_call_and_return_conditional_losses_12048w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�O
�
D__inference_MNIST_CNN_layer_call_and_return_conditional_losses_12518

inputs
quantize_layer_12425: 
quantize_layer_12427: +
quant_conv1_12430:
quant_conv1_12432:
quant_conv1_12434:
quant_conv1_12436:
quant_bn1_12439:
quant_bn1_12441:
quant_bn1_12443:
quant_bn1_12445:
quant_relu1_12448: 
quant_relu1_12450: +
quant_conv2_12454:
quant_conv2_12456:
quant_conv2_12458:
quant_conv2_12460:
quant_bn2_12463:
quant_bn2_12465:
quant_bn2_12467:
quant_bn2_12469:
quant_relu2_12472: 
quant_relu2_12474: +
quant_conv3_12478:
quant_conv3_12480:
quant_conv3_12482:
quant_conv3_12484:
quant_bn3_12487:
quant_bn3_12489:
quant_bn3_12491:
quant_bn3_12493:
quant_relu3_12496: 
quant_relu3_12498: %
quant_dense1_12503:	�

quant_dense1_12505: 
quant_dense1_12507:  
quant_dense1_12509:

quant_dense1_12511: 
quant_dense1_12513: 
identity��!quant_bn1/StatefulPartitionedCall�!quant_bn2/StatefulPartitionedCall�!quant_bn3/StatefulPartitionedCall�#quant_conv1/StatefulPartitionedCall�#quant_conv2/StatefulPartitionedCall�#quant_conv3/StatefulPartitionedCall�$quant_dense1/StatefulPartitionedCall�#quant_relu1/StatefulPartitionedCall�#quant_relu2/StatefulPartitionedCall�#quant_relu3/StatefulPartitionedCall�&quantize_layer/StatefulPartitionedCall�
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_12425quantize_layer_12427*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quantize_layer_layer_call_and_return_conditional_losses_11743�
#quant_conv1/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv1_12430quant_conv1_12432quant_conv1_12434quant_conv1_12436*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_conv1_layer_call_and_return_conditional_losses_11776�
!quant_bn1/StatefulPartitionedCallStatefulPartitionedCall,quant_conv1/StatefulPartitionedCall:output:0quant_bn1_12439quant_bn1_12441quant_bn1_12443quant_bn1_12445*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_quant_bn1_layer_call_and_return_conditional_losses_11804�
#quant_relu1/StatefulPartitionedCallStatefulPartitionedCall*quant_bn1/StatefulPartitionedCall:output:0quant_relu1_12448quant_relu1_12450*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_relu1_layer_call_and_return_conditional_losses_11840�
quant_maxpool1/PartitionedCallPartitionedCall,quant_relu1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_maxpool1_layer_call_and_return_conditional_losses_11851�
#quant_conv2/StatefulPartitionedCallStatefulPartitionedCall'quant_maxpool1/PartitionedCall:output:0quant_conv2_12454quant_conv2_12456quant_conv2_12458quant_conv2_12460*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_conv2_layer_call_and_return_conditional_losses_11880�
!quant_bn2/StatefulPartitionedCallStatefulPartitionedCall,quant_conv2/StatefulPartitionedCall:output:0quant_bn2_12463quant_bn2_12465quant_bn2_12467quant_bn2_12469*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_quant_bn2_layer_call_and_return_conditional_losses_11908�
#quant_relu2/StatefulPartitionedCallStatefulPartitionedCall*quant_bn2/StatefulPartitionedCall:output:0quant_relu2_12472quant_relu2_12474*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_relu2_layer_call_and_return_conditional_losses_11944�
quant_maxpool2/PartitionedCallPartitionedCall,quant_relu2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_maxpool2_layer_call_and_return_conditional_losses_11955�
#quant_conv3/StatefulPartitionedCallStatefulPartitionedCall'quant_maxpool2/PartitionedCall:output:0quant_conv3_12478quant_conv3_12480quant_conv3_12482quant_conv3_12484*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_conv3_layer_call_and_return_conditional_losses_11984�
!quant_bn3/StatefulPartitionedCallStatefulPartitionedCall,quant_conv3/StatefulPartitionedCall:output:0quant_bn3_12487quant_bn3_12489quant_bn3_12491quant_bn3_12493*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_quant_bn3_layer_call_and_return_conditional_losses_12012�
#quant_relu3/StatefulPartitionedCallStatefulPartitionedCall*quant_bn3/StatefulPartitionedCall:output:0quant_relu3_12496quant_relu3_12498*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_relu3_layer_call_and_return_conditional_losses_12048�
quant_maxpool3/PartitionedCallPartitionedCall,quant_relu3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_maxpool3_layer_call_and_return_conditional_losses_12059�
quant_flatten/PartitionedCallPartitionedCall'quant_maxpool3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_quant_flatten_layer_call_and_return_conditional_losses_12067�
$quant_dense1/StatefulPartitionedCallStatefulPartitionedCall&quant_flatten/PartitionedCall:output:0quant_dense1_12503quant_dense1_12505quant_dense1_12507quant_dense1_12509quant_dense1_12511quant_dense1_12513*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_quant_dense1_layer_call_and_return_conditional_losses_12117�
quant_softmax/PartitionedCallPartitionedCall-quant_dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_quant_softmax_layer_call_and_return_conditional_losses_12136u
IdentityIdentity&quant_softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp"^quant_bn1/StatefulPartitionedCall"^quant_bn2/StatefulPartitionedCall"^quant_bn3/StatefulPartitionedCall$^quant_conv1/StatefulPartitionedCall$^quant_conv2/StatefulPartitionedCall$^quant_conv3/StatefulPartitionedCall%^quant_dense1/StatefulPartitionedCall$^quant_relu1/StatefulPartitionedCall$^quant_relu2/StatefulPartitionedCall$^quant_relu3/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!quant_bn1/StatefulPartitionedCall!quant_bn1/StatefulPartitionedCall2F
!quant_bn2/StatefulPartitionedCall!quant_bn2/StatefulPartitionedCall2F
!quant_bn3/StatefulPartitionedCall!quant_bn3/StatefulPartitionedCall2J
#quant_conv1/StatefulPartitionedCall#quant_conv1/StatefulPartitionedCall2J
#quant_conv2/StatefulPartitionedCall#quant_conv2/StatefulPartitionedCall2J
#quant_conv3/StatefulPartitionedCall#quant_conv3/StatefulPartitionedCall2L
$quant_dense1/StatefulPartitionedCall$quant_dense1/StatefulPartitionedCall2J
#quant_relu1/StatefulPartitionedCall#quant_relu1/StatefulPartitionedCall2J
#quant_relu2/StatefulPartitionedCall#quant_relu2/StatefulPartitionedCall2J
#quant_relu3/StatefulPartitionedCall#quant_relu3/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�*
�
F__inference_quant_relu2_layer_call_and_return_conditional_losses_14309

inputs@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1N
ReluReluinputs*
T0*/
_output_shapes
:���������p
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
,__inference_quant_dense1_layer_call_fn_14599

inputs
unknown:	�

	unknown_0: 
	unknown_1: 
	unknown_2:

	unknown_3: 
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_quant_dense1_layer_call_and_return_conditional_losses_12398o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
D__inference_quant_bn2_layer_call_and_return_conditional_losses_12269

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_quantize_layer_layer_call_and_return_conditional_losses_13931

inputsK
Aallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: M
Callvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCallvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:���������  �
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:���������  �
NoOpNoOp9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
F__inference_quant_relu1_layer_call_and_return_conditional_losses_14115

inputsK
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1N
ReluReluinputs*
T0*/
_output_shapes
:���������  �
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:���������  �
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:���������  �
NoOpNoOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
>__inference_bn2_layer_call_and_return_conditional_losses_14821

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
+__inference_quant_conv2_layer_call_fn_14161

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_conv2_layer_call_and_return_conditional_losses_12242w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_quant_conv1_layer_call_fn_13957

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_conv1_layer_call_and_return_conditional_losses_12170w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������  : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
F__inference_quant_conv2_layer_call_and_return_conditional_losses_14203

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0�
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(�
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
d
H__inference_quant_softmax_layer_call_and_return_conditional_losses_12416

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:���������
Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������
:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
_
C__inference_maxpool2_layer_call_and_return_conditional_losses_14831

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�P
�
D__inference_MNIST_CNN_layer_call_and_return_conditional_losses_12419	
input
quantize_layer_12151: 
quantize_layer_12153: +
quant_conv1_12171:
quant_conv1_12173:
quant_conv1_12175:
quant_conv1_12177:
quant_bn1_12198:
quant_bn1_12200:
quant_bn1_12202:
quant_bn1_12204:
quant_relu1_12217: 
quant_relu1_12219: +
quant_conv2_12243:
quant_conv2_12245:
quant_conv2_12247:
quant_conv2_12249:
quant_bn2_12270:
quant_bn2_12272:
quant_bn2_12274:
quant_bn2_12276:
quant_relu2_12289: 
quant_relu2_12291: +
quant_conv3_12315:
quant_conv3_12317:
quant_conv3_12319:
quant_conv3_12321:
quant_bn3_12342:
quant_bn3_12344:
quant_bn3_12346:
quant_bn3_12348:
quant_relu3_12361: 
quant_relu3_12363: %
quant_dense1_12399:	�

quant_dense1_12401: 
quant_dense1_12403:  
quant_dense1_12405:

quant_dense1_12407: 
quant_dense1_12409: 
identity��!quant_bn1/StatefulPartitionedCall�!quant_bn2/StatefulPartitionedCall�!quant_bn3/StatefulPartitionedCall�#quant_conv1/StatefulPartitionedCall�#quant_conv2/StatefulPartitionedCall�#quant_conv3/StatefulPartitionedCall�$quant_dense1/StatefulPartitionedCall�#quant_relu1/StatefulPartitionedCall�#quant_relu2/StatefulPartitionedCall�#quant_relu3/StatefulPartitionedCall�&quantize_layer/StatefulPartitionedCall�
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallinputquantize_layer_12151quantize_layer_12153*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quantize_layer_layer_call_and_return_conditional_losses_12150�
#quant_conv1/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv1_12171quant_conv1_12173quant_conv1_12175quant_conv1_12177*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_conv1_layer_call_and_return_conditional_losses_12170�
!quant_bn1/StatefulPartitionedCallStatefulPartitionedCall,quant_conv1/StatefulPartitionedCall:output:0quant_bn1_12198quant_bn1_12200quant_bn1_12202quant_bn1_12204*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_quant_bn1_layer_call_and_return_conditional_losses_12197�
#quant_relu1/StatefulPartitionedCallStatefulPartitionedCall*quant_bn1/StatefulPartitionedCall:output:0quant_relu1_12217quant_relu1_12219*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_relu1_layer_call_and_return_conditional_losses_12216�
quant_maxpool1/PartitionedCallPartitionedCall,quant_relu1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_maxpool1_layer_call_and_return_conditional_losses_12226�
#quant_conv2/StatefulPartitionedCallStatefulPartitionedCall'quant_maxpool1/PartitionedCall:output:0quant_conv2_12243quant_conv2_12245quant_conv2_12247quant_conv2_12249*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_conv2_layer_call_and_return_conditional_losses_12242�
!quant_bn2/StatefulPartitionedCallStatefulPartitionedCall,quant_conv2/StatefulPartitionedCall:output:0quant_bn2_12270quant_bn2_12272quant_bn2_12274quant_bn2_12276*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_quant_bn2_layer_call_and_return_conditional_losses_12269�
#quant_relu2/StatefulPartitionedCallStatefulPartitionedCall*quant_bn2/StatefulPartitionedCall:output:0quant_relu2_12289quant_relu2_12291*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_relu2_layer_call_and_return_conditional_losses_12288�
quant_maxpool2/PartitionedCallPartitionedCall,quant_relu2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_maxpool2_layer_call_and_return_conditional_losses_12298�
#quant_conv3/StatefulPartitionedCallStatefulPartitionedCall'quant_maxpool2/PartitionedCall:output:0quant_conv3_12315quant_conv3_12317quant_conv3_12319quant_conv3_12321*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_conv3_layer_call_and_return_conditional_losses_12314�
!quant_bn3/StatefulPartitionedCallStatefulPartitionedCall,quant_conv3/StatefulPartitionedCall:output:0quant_bn3_12342quant_bn3_12344quant_bn3_12346quant_bn3_12348*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_quant_bn3_layer_call_and_return_conditional_losses_12341�
#quant_relu3/StatefulPartitionedCallStatefulPartitionedCall*quant_bn3/StatefulPartitionedCall:output:0quant_relu3_12361quant_relu3_12363*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_relu3_layer_call_and_return_conditional_losses_12360�
quant_maxpool3/PartitionedCallPartitionedCall,quant_relu3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_maxpool3_layer_call_and_return_conditional_losses_12370�
quant_flatten/PartitionedCallPartitionedCall'quant_maxpool3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_quant_flatten_layer_call_and_return_conditional_losses_12377�
$quant_dense1/StatefulPartitionedCallStatefulPartitionedCall&quant_flatten/PartitionedCall:output:0quant_dense1_12399quant_dense1_12401quant_dense1_12403quant_dense1_12405quant_dense1_12407quant_dense1_12409*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_quant_dense1_layer_call_and_return_conditional_losses_12398�
quant_softmax/PartitionedCallPartitionedCall-quant_dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_quant_softmax_layer_call_and_return_conditional_losses_12416u
IdentityIdentity&quant_softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp"^quant_bn1/StatefulPartitionedCall"^quant_bn2/StatefulPartitionedCall"^quant_bn3/StatefulPartitionedCall$^quant_conv1/StatefulPartitionedCall$^quant_conv2/StatefulPartitionedCall$^quant_conv3/StatefulPartitionedCall%^quant_dense1/StatefulPartitionedCall$^quant_relu1/StatefulPartitionedCall$^quant_relu2/StatefulPartitionedCall$^quant_relu3/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!quant_bn1/StatefulPartitionedCall!quant_bn1/StatefulPartitionedCall2F
!quant_bn2/StatefulPartitionedCall!quant_bn2/StatefulPartitionedCall2F
!quant_bn3/StatefulPartitionedCall!quant_bn3/StatefulPartitionedCall2J
#quant_conv1/StatefulPartitionedCall#quant_conv1/StatefulPartitionedCall2J
#quant_conv2/StatefulPartitionedCall#quant_conv2/StatefulPartitionedCall2J
#quant_conv3/StatefulPartitionedCall#quant_conv3/StatefulPartitionedCall2L
$quant_dense1/StatefulPartitionedCall$quant_dense1/StatefulPartitionedCall2J
#quant_relu1/StatefulPartitionedCall#quant_relu1/StatefulPartitionedCall2J
#quant_relu2/StatefulPartitionedCall#quant_relu2/StatefulPartitionedCall2J
#quant_relu3/StatefulPartitionedCall#quant_relu3/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:V R
/
_output_shapes
:���������  

_user_specified_nameinput
�
e
I__inference_quant_maxpool3_layer_call_and_return_conditional_losses_12059

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�*
�
F__inference_quant_relu2_layer_call_and_return_conditional_losses_11944

inputs@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1N
ReluReluinputs*
T0*/
_output_shapes
:���������p
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
D
(__inference_maxpool1_layer_call_fn_14754

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_maxpool1_layer_call_and_return_conditional_losses_11560�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
>__inference_bn1_layer_call_and_return_conditional_losses_14749

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
>__inference_bn1_layer_call_and_return_conditional_losses_14731

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
F__inference_quant_relu3_layer_call_and_return_conditional_losses_12360

inputsK
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1N
ReluReluinputs*
T0*/
_output_shapes
:����������
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_quant_relu1_layer_call_and_return_conditional_losses_12216

inputsK
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1N
ReluReluinputs*
T0*/
_output_shapes
:���������  �
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:���������  �
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:���������  �
NoOpNoOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
d
H__inference_quant_softmax_layer_call_and_return_conditional_losses_14687

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:���������
Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������
:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
>__inference_bn2_layer_call_and_return_conditional_losses_11585

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�*
�
F__inference_quant_relu3_layer_call_and_return_conditional_losses_14513

inputs@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1N
ReluReluinputs*
T0*/
_output_shapes
:���������p
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
>__inference_bn3_layer_call_and_return_conditional_losses_14893

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
)__inference_quant_bn1_layer_call_fn_14025

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_quant_bn1_layer_call_and_return_conditional_losses_12197w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������  : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
D__inference_quant_bn3_layer_call_and_return_conditional_losses_12341

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�,
�
F__inference_quant_conv2_layer_call_and_return_conditional_losses_14188

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2�
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:�
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:�
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:�
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:�
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0�
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(�
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�,
�
F__inference_quant_conv3_layer_call_and_return_conditional_losses_14392

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2�
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:�
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:�
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:�
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:�
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0�
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(�
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
>__inference_bn3_layer_call_and_return_conditional_losses_11679

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
)__inference_quant_bn2_layer_call_fn_14216

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_quant_bn2_layer_call_and_return_conditional_losses_11908w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_quant_bn3_layer_call_and_return_conditional_losses_12012

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
#__inference_bn3_layer_call_fn_14857

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn3_layer_call_and_return_conditional_losses_11679�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
+__inference_quant_relu1_layer_call_fn_14070

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_quant_relu1_layer_call_and_return_conditional_losses_11840w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
e
I__inference_quant_maxpool2_layer_call_and_return_conditional_losses_14334

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
,__inference_quant_dense1_layer_call_fn_14582

inputs
unknown:	�

	unknown_0: 
	unknown_1: 
	unknown_2:

	unknown_3: 
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_quant_dense1_layer_call_and_return_conditional_losses_12117o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
I__inference_quant_maxpool2_layer_call_and_return_conditional_losses_14339

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_quant_conv3_layer_call_and_return_conditional_losses_12314

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1�ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0�
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(�
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2�
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12�
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
e
I__inference_quant_maxpool1_layer_call_and_return_conditional_losses_14135

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  :W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
.__inference_quantize_layer_layer_call_fn_13901

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quantize_layer_layer_call_and_return_conditional_losses_12150w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
>__inference_bn1_layer_call_and_return_conditional_losses_11509

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
input6
serving_default_input:0���������  A
quant_softmax0
StatefulPartitionedCall:0���������
tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer_with_weights-10
layer-11
layer_with_weights-11
layer-12
layer_with_weights-12
layer-13
layer_with_weights-13
layer-14
layer_with_weights-14
layer-15
layer_with_weights-15
layer-16
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses
!quantize_layer_min
"quantize_layer_max
#quantizer_vars
$optimizer_step"
_tf_keras_layer
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
	+layer
,optimizer_step
-_weight_vars
.
kernel_min
/
kernel_max
0_quantize_activations
1_output_quantizers"
_tf_keras_layer
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
	8layer
9optimizer_step
:_weight_vars
;_quantize_activations
<_output_quantizers"
_tf_keras_layer
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
	Clayer
Doptimizer_step
E_weight_vars
F_quantize_activations
G_output_quantizers
H
output_min
I
output_max
J_output_quantizer_vars"
_tf_keras_layer
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses
	Qlayer
Roptimizer_step
S_weight_vars
T_quantize_activations
U_output_quantizers"
_tf_keras_layer
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses
	\layer
]optimizer_step
^_weight_vars
_
kernel_min
`
kernel_max
a_quantize_activations
b_output_quantizers"
_tf_keras_layer
�
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses
	ilayer
joptimizer_step
k_weight_vars
l_quantize_activations
m_output_quantizers"
_tf_keras_layer
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses
	tlayer
uoptimizer_step
v_weight_vars
w_quantize_activations
x_output_quantizers
y
output_min
z
output_max
{_output_quantizer_vars"
_tf_keras_layer
�
|	variables
}trainable_variables
~regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�layer
�optimizer_step
�_weight_vars
�_quantize_activations
�_output_quantizers"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�layer
�optimizer_step
�_weight_vars
�
kernel_min
�
kernel_max
�_quantize_activations
�_output_quantizers"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�layer
�optimizer_step
�_weight_vars
�_quantize_activations
�_output_quantizers"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�layer
�optimizer_step
�_weight_vars
�_quantize_activations
�_output_quantizers
�
output_min
�
output_max
�_output_quantizer_vars"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�layer
�optimizer_step
�_weight_vars
�_quantize_activations
�_output_quantizers"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�layer
�optimizer_step
�_weight_vars
�_quantize_activations
�_output_quantizers"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�layer
�optimizer_step
�_weight_vars
�
kernel_min
�
kernel_max
�_quantize_activations
�post_activation_min
�post_activation_max
�_output_quantizers"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�layer
�optimizer_step
�_weight_vars
�_quantize_activations
�_output_quantizers"
_tf_keras_layer
�
!0
"1
$2
�3
�4
,5
.6
/7
�8
�9
�10
�11
912
D13
H14
I15
R16
�17
�18
]19
_20
`21
�22
�23
�24
�25
j26
u27
y28
z29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
)__inference_MNIST_CNN_layer_call_fn_12597
)__inference_MNIST_CNN_layer_call_fn_12774
)__inference_MNIST_CNN_layer_call_fn_13430
)__inference_MNIST_CNN_layer_call_fn_13511�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
D__inference_MNIST_CNN_layer_call_and_return_conditional_losses_12139
D__inference_MNIST_CNN_layer_call_and_return_conditional_losses_12419
D__inference_MNIST_CNN_layer_call_and_return_conditional_losses_13759
D__inference_MNIST_CNN_layer_call_and_return_conditional_losses_13883�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
 __inference__wrapped_model_11490input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
5
!0
"1
$2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
.__inference_quantize_layer_layer_call_fn_13892
.__inference_quantize_layer_layer_call_fn_13901�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
I__inference_quantize_layer_layer_call_and_return_conditional_losses_13922
I__inference_quantize_layer_layer_call_and_return_conditional_losses_13931�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
):' 2!quantize_layer/quantize_layer_min
):' 2!quantize_layer/quantize_layer_max
:
!min_var
"max_var"
trackable_dict_wrapper
%:# 2quantize_layer/optimizer_step
E
�0
�1
,2
.3
/4"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_quant_conv1_layer_call_fn_13944
+__inference_quant_conv1_layer_call_fn_13957�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_quant_conv1_layer_call_and_return_conditional_losses_13984
F__inference_quant_conv1_layer_call_and_return_conditional_losses_13999�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
":  2quant_conv1/optimizer_step
(
�0"
trackable_list_wrapper
": 2quant_conv1/kernel_min
": 2quant_conv1/kernel_max
 "
trackable_list_wrapper
 "
trackable_list_wrapper
G
�0
�1
�2
�3
94"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
)__inference_quant_bn1_layer_call_fn_14012
)__inference_quant_bn1_layer_call_fn_14025�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
D__inference_quant_bn1_layer_call_and_return_conditional_losses_14043
D__inference_quant_bn1_layer_call_and_return_conditional_losses_14061�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
 : 2quant_bn1/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
D0
H1
I2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_quant_relu1_layer_call_fn_14070
+__inference_quant_relu1_layer_call_fn_14079�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_quant_relu1_layer_call_and_return_conditional_losses_14105
F__inference_quant_relu1_layer_call_and_return_conditional_losses_14115�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
":  2quant_relu1/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
: 2quant_relu1/output_min
: 2quant_relu1/output_max
:
Hmin_var
Imax_var"
trackable_dict_wrapper
'
R0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
.__inference_quant_maxpool1_layer_call_fn_14120
.__inference_quant_maxpool1_layer_call_fn_14125�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
I__inference_quant_maxpool1_layer_call_and_return_conditional_losses_14130
I__inference_quant_maxpool1_layer_call_and_return_conditional_losses_14135�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
%:# 2quant_maxpool1/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
E
�0
�1
]2
_3
`4"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_quant_conv2_layer_call_fn_14148
+__inference_quant_conv2_layer_call_fn_14161�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_quant_conv2_layer_call_and_return_conditional_losses_14188
F__inference_quant_conv2_layer_call_and_return_conditional_losses_14203�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
":  2quant_conv2/optimizer_step
(
�0"
trackable_list_wrapper
": 2quant_conv2/kernel_min
": 2quant_conv2/kernel_max
 "
trackable_list_wrapper
 "
trackable_list_wrapper
G
�0
�1
�2
�3
j4"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
)__inference_quant_bn2_layer_call_fn_14216
)__inference_quant_bn2_layer_call_fn_14229�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
D__inference_quant_bn2_layer_call_and_return_conditional_losses_14247
D__inference_quant_bn2_layer_call_and_return_conditional_losses_14265�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
 : 2quant_bn2/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
u0
y1
z2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_quant_relu2_layer_call_fn_14274
+__inference_quant_relu2_layer_call_fn_14283�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_quant_relu2_layer_call_and_return_conditional_losses_14309
F__inference_quant_relu2_layer_call_and_return_conditional_losses_14319�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
":  2quant_relu2/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
: 2quant_relu2/output_min
: 2quant_relu2/output_max
:
ymin_var
zmax_var"
trackable_dict_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
|	variables
}trainable_variables
~regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
.__inference_quant_maxpool2_layer_call_fn_14324
.__inference_quant_maxpool2_layer_call_fn_14329�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
I__inference_quant_maxpool2_layer_call_and_return_conditional_losses_14334
I__inference_quant_maxpool2_layer_call_and_return_conditional_losses_14339�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
%:# 2quant_maxpool2/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_quant_conv3_layer_call_fn_14352
+__inference_quant_conv3_layer_call_fn_14365�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_quant_conv3_layer_call_and_return_conditional_losses_14392
F__inference_quant_conv3_layer_call_and_return_conditional_losses_14407�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
":  2quant_conv3/optimizer_step
(
�0"
trackable_list_wrapper
": 2quant_conv3/kernel_min
": 2quant_conv3/kernel_max
 "
trackable_list_wrapper
 "
trackable_list_wrapper
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
)__inference_quant_bn3_layer_call_fn_14420
)__inference_quant_bn3_layer_call_fn_14433�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
D__inference_quant_bn3_layer_call_and_return_conditional_losses_14451
D__inference_quant_bn3_layer_call_and_return_conditional_losses_14469�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
 : 2quant_bn3/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_quant_relu3_layer_call_fn_14478
+__inference_quant_relu3_layer_call_fn_14487�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_quant_relu3_layer_call_and_return_conditional_losses_14513
F__inference_quant_relu3_layer_call_and_return_conditional_losses_14523�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
":  2quant_relu3/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
: 2quant_relu3/output_min
: 2quant_relu3/output_max
<
�min_var
�max_var"
trackable_dict_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
.__inference_quant_maxpool3_layer_call_fn_14528
.__inference_quant_maxpool3_layer_call_fn_14533�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
I__inference_quant_maxpool3_layer_call_and_return_conditional_losses_14538
I__inference_quant_maxpool3_layer_call_and_return_conditional_losses_14543�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
%:# 2quant_maxpool3/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
-__inference_quant_flatten_layer_call_fn_14548
-__inference_quant_flatten_layer_call_fn_14553�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
H__inference_quant_flatten_layer_call_and_return_conditional_losses_14559
H__inference_quant_flatten_layer_call_and_return_conditional_losses_14565�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
$:" 2quant_flatten/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
�0
�1
�2
�3
�4
�5
�6"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_quant_dense1_layer_call_fn_14582
,__inference_quant_dense1_layer_call_fn_14599�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_quant_dense1_layer_call_and_return_conditional_losses_14647
G__inference_quant_dense1_layer_call_and_return_conditional_losses_14667�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
#:! 2quant_dense1/optimizer_step
(
�0"
trackable_list_wrapper
: 2quant_dense1/kernel_min
: 2quant_dense1/kernel_max
 "
trackable_list_wrapper
(:& 2 quant_dense1/post_activation_min
(:& 2 quant_dense1/post_activation_max
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
-__inference_quant_softmax_layer_call_fn_14672
-__inference_quant_softmax_layer_call_fn_14677�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
H__inference_quant_softmax_layer_call_and_return_conditional_losses_14682
H__inference_quant_softmax_layer_call_and_return_conditional_losses_14687�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
$:" 2quant_softmax/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
&:$2conv1/kernel
:2
conv1/bias
:2	bn1/gamma
:2bn1/beta
: (2bn1/moving_mean
#:! (2bn1/moving_variance
&:$2conv2/kernel
:2
conv2/bias
:2	bn2/gamma
:2bn2/beta
: (2bn2/moving_mean
#:! (2bn2/moving_variance
&:$2conv3/kernel
:2
conv3/bias
:2	bn3/gamma
:2bn3/beta
: (2bn3/moving_mean
#:! (2bn3/moving_variance
 :	�
2dense1/kernel
:
2dense1/bias
�
!0
"1
$2
,3
.4
/5
�6
�7
98
D9
H10
I11
R12
]13
_14
`15
�16
�17
j18
u19
y20
z21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_MNIST_CNN_layer_call_fn_12597input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_MNIST_CNN_layer_call_fn_12774input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_MNIST_CNN_layer_call_fn_13430inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_MNIST_CNN_layer_call_fn_13511inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_MNIST_CNN_layer_call_and_return_conditional_losses_12139input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_MNIST_CNN_layer_call_and_return_conditional_losses_12419input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_MNIST_CNN_layer_call_and_return_conditional_losses_13759inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_MNIST_CNN_layer_call_and_return_conditional_losses_13883inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
#__inference_signature_wrapper_13349input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
5
!0
"1
$2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_quantize_layer_layer_call_fn_13892inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_quantize_layer_layer_call_fn_13901inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_quantize_layer_layer_call_and_return_conditional_losses_13922inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_quantize_layer_layer_call_and_return_conditional_losses_13931inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
5
,0
.1
/2"
trackable_list_wrapper
'
+0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_quant_conv1_layer_call_fn_13944inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_quant_conv1_layer_call_fn_13957inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_quant_conv1_layer_call_and_return_conditional_losses_13984inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_quant_conv1_layer_call_and_return_conditional_losses_13999inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
1
�0
�2"
trackable_tuple_wrapper
7
�0
�1
92"
trackable_list_wrapper
'
80"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_quant_bn1_layer_call_fn_14012inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_quant_bn1_layer_call_fn_14025inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_quant_bn1_layer_call_and_return_conditional_losses_14043inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_quant_bn1_layer_call_and_return_conditional_losses_14061inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
#__inference_bn1_layer_call_fn_14700
#__inference_bn1_layer_call_fn_14713�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
>__inference_bn1_layer_call_and_return_conditional_losses_14731
>__inference_bn1_layer_call_and_return_conditional_losses_14749�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
5
D0
H1
I2"
trackable_list_wrapper
'
C0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_quant_relu1_layer_call_fn_14070inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_quant_relu1_layer_call_fn_14079inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_quant_relu1_layer_call_and_return_conditional_losses_14105inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_quant_relu1_layer_call_and_return_conditional_losses_14115inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
R0"
trackable_list_wrapper
'
Q0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_quant_maxpool1_layer_call_fn_14120inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_quant_maxpool1_layer_call_fn_14125inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_quant_maxpool1_layer_call_and_return_conditional_losses_14130inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_quant_maxpool1_layer_call_and_return_conditional_losses_14135inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_maxpool1_layer_call_fn_14754�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_maxpool1_layer_call_and_return_conditional_losses_14759�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
5
]0
_1
`2"
trackable_list_wrapper
'
\0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_quant_conv2_layer_call_fn_14148inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_quant_conv2_layer_call_fn_14161inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_quant_conv2_layer_call_and_return_conditional_losses_14188inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_quant_conv2_layer_call_and_return_conditional_losses_14203inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
1
�0
�2"
trackable_tuple_wrapper
7
�0
�1
j2"
trackable_list_wrapper
'
i0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_quant_bn2_layer_call_fn_14216inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_quant_bn2_layer_call_fn_14229inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_quant_bn2_layer_call_and_return_conditional_losses_14247inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_quant_bn2_layer_call_and_return_conditional_losses_14265inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
#__inference_bn2_layer_call_fn_14772
#__inference_bn2_layer_call_fn_14785�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
>__inference_bn2_layer_call_and_return_conditional_losses_14803
>__inference_bn2_layer_call_and_return_conditional_losses_14821�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
5
u0
y1
z2"
trackable_list_wrapper
'
t0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_quant_relu2_layer_call_fn_14274inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_quant_relu2_layer_call_fn_14283inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_quant_relu2_layer_call_and_return_conditional_losses_14309inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_quant_relu2_layer_call_and_return_conditional_losses_14319inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_quant_maxpool2_layer_call_fn_14324inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_quant_maxpool2_layer_call_fn_14329inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_quant_maxpool2_layer_call_and_return_conditional_losses_14334inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_quant_maxpool2_layer_call_and_return_conditional_losses_14339inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_maxpool2_layer_call_fn_14826�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_maxpool2_layer_call_and_return_conditional_losses_14831�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
8
�0
�1
�2"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_quant_conv3_layer_call_fn_14352inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_quant_conv3_layer_call_fn_14365inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_quant_conv3_layer_call_and_return_conditional_losses_14392inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_quant_conv3_layer_call_and_return_conditional_losses_14407inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
1
�0
�2"
trackable_tuple_wrapper
8
�0
�1
�2"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_quant_bn3_layer_call_fn_14420inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_quant_bn3_layer_call_fn_14433inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_quant_bn3_layer_call_and_return_conditional_losses_14451inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_quant_bn3_layer_call_and_return_conditional_losses_14469inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
#__inference_bn3_layer_call_fn_14844
#__inference_bn3_layer_call_fn_14857�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
>__inference_bn3_layer_call_and_return_conditional_losses_14875
>__inference_bn3_layer_call_and_return_conditional_losses_14893�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_quant_relu3_layer_call_fn_14478inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_quant_relu3_layer_call_fn_14487inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_quant_relu3_layer_call_and_return_conditional_losses_14513inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_quant_relu3_layer_call_and_return_conditional_losses_14523inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_quant_maxpool3_layer_call_fn_14528inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_quant_maxpool3_layer_call_fn_14533inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_quant_maxpool3_layer_call_and_return_conditional_losses_14538inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_quant_maxpool3_layer_call_and_return_conditional_losses_14543inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_maxpool3_layer_call_fn_14898�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_maxpool3_layer_call_and_return_conditional_losses_14903�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_quant_flatten_layer_call_fn_14548inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_quant_flatten_layer_call_fn_14553inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_quant_flatten_layer_call_and_return_conditional_losses_14559inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_quant_flatten_layer_call_and_return_conditional_losses_14565inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_quant_dense1_layer_call_fn_14582inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_quant_dense1_layer_call_fn_14599inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_quant_dense1_layer_call_and_return_conditional_losses_14647inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_quant_dense1_layer_call_and_return_conditional_losses_14667inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
1
�0
�2"
trackable_tuple_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_quant_softmax_layer_call_fn_14672inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_quant_softmax_layer_call_fn_14677inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_quant_softmax_layer_call_and_return_conditional_losses_14682inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_quant_softmax_layer_call_and_return_conditional_losses_14687inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
+:)2Adam/m/conv1/kernel
+:)2Adam/v/conv1/kernel
:2Adam/m/conv1/bias
:2Adam/v/conv1/bias
:2Adam/m/bn1/gamma
:2Adam/v/bn1/gamma
:2Adam/m/bn1/beta
:2Adam/v/bn1/beta
+:)2Adam/m/conv2/kernel
+:)2Adam/v/conv2/kernel
:2Adam/m/conv2/bias
:2Adam/v/conv2/bias
:2Adam/m/bn2/gamma
:2Adam/v/bn2/gamma
:2Adam/m/bn2/beta
:2Adam/v/bn2/beta
+:)2Adam/m/conv3/kernel
+:)2Adam/v/conv3/kernel
:2Adam/m/conv3/bias
:2Adam/v/conv3/bias
:2Adam/m/bn3/gamma
:2Adam/v/bn3/gamma
:2Adam/m/bn3/beta
:2Adam/v/bn3/beta
%:#	�
2Adam/m/dense1/kernel
%:#	�
2Adam/v/dense1/kernel
:
2Adam/m/dense1/bias
:
2Adam/v/dense1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:
.min_var
/max_var"
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_bn1_layer_call_fn_14700inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_bn1_layer_call_fn_14713inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_bn1_layer_call_and_return_conditional_losses_14731inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_bn1_layer_call_and_return_conditional_losses_14749inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_maxpool1_layer_call_fn_14754inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_maxpool1_layer_call_and_return_conditional_losses_14759inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:
_min_var
`max_var"
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_bn2_layer_call_fn_14772inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_bn2_layer_call_fn_14785inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_bn2_layer_call_and_return_conditional_losses_14803inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_bn2_layer_call_and_return_conditional_losses_14821inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_maxpool2_layer_call_fn_14826inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_maxpool2_layer_call_and_return_conditional_losses_14831inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
�min_var
�max_var"
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_bn3_layer_call_fn_14844inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_bn3_layer_call_fn_14857inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_bn3_layer_call_and_return_conditional_losses_14875inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_bn3_layer_call_and_return_conditional_losses_14893inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_maxpool3_layer_call_fn_14898inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_maxpool3_layer_call_and_return_conditional_losses_14903inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
�min_var
�max_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
D__inference_MNIST_CNN_layer_call_and_return_conditional_losses_12139�B!"�./�����HI�_`�����yz����������������>�;
4�1
'�$
input���������  
p

 
� ",�)
"�
tensor_0���������

� �
D__inference_MNIST_CNN_layer_call_and_return_conditional_losses_12419�B!"�./�����HI�_`�����yz����������������>�;
4�1
'�$
input���������  
p 

 
� ",�)
"�
tensor_0���������

� �
D__inference_MNIST_CNN_layer_call_and_return_conditional_losses_13759�B!"�./�����HI�_`�����yz����������������?�<
5�2
(�%
inputs���������  
p

 
� ",�)
"�
tensor_0���������

� �
D__inference_MNIST_CNN_layer_call_and_return_conditional_losses_13883�B!"�./�����HI�_`�����yz����������������?�<
5�2
(�%
inputs���������  
p 

 
� ",�)
"�
tensor_0���������

� �
)__inference_MNIST_CNN_layer_call_fn_12597�B!"�./�����HI�_`�����yz����������������>�;
4�1
'�$
input���������  
p

 
� "!�
unknown���������
�
)__inference_MNIST_CNN_layer_call_fn_12774�B!"�./�����HI�_`�����yz����������������>�;
4�1
'�$
input���������  
p 

 
� "!�
unknown���������
�
)__inference_MNIST_CNN_layer_call_fn_13430�B!"�./�����HI�_`�����yz����������������?�<
5�2
(�%
inputs���������  
p

 
� "!�
unknown���������
�
)__inference_MNIST_CNN_layer_call_fn_13511�B!"�./�����HI�_`�����yz����������������?�<
5�2
(�%
inputs���������  
p 

 
� "!�
unknown���������
�
 __inference__wrapped_model_11490�B!"�./�����HI�_`�����yz����������������6�3
,�)
'�$
input���������  
� "=�:
8
quant_softmax'�$
quant_softmax���������
�
>__inference_bn1_layer_call_and_return_conditional_losses_14731�����Q�N
G�D
:�7
inputs+���������������������������
p

 
� "F�C
<�9
tensor_0+���������������������������
� �
>__inference_bn1_layer_call_and_return_conditional_losses_14749�����Q�N
G�D
:�7
inputs+���������������������������
p 

 
� "F�C
<�9
tensor_0+���������������������������
� �
#__inference_bn1_layer_call_fn_14700�����Q�N
G�D
:�7
inputs+���������������������������
p

 
� ";�8
unknown+����������������������������
#__inference_bn1_layer_call_fn_14713�����Q�N
G�D
:�7
inputs+���������������������������
p 

 
� ";�8
unknown+����������������������������
>__inference_bn2_layer_call_and_return_conditional_losses_14803�����Q�N
G�D
:�7
inputs+���������������������������
p

 
� "F�C
<�9
tensor_0+���������������������������
� �
>__inference_bn2_layer_call_and_return_conditional_losses_14821�����Q�N
G�D
:�7
inputs+���������������������������
p 

 
� "F�C
<�9
tensor_0+���������������������������
� �
#__inference_bn2_layer_call_fn_14772�����Q�N
G�D
:�7
inputs+���������������������������
p

 
� ";�8
unknown+����������������������������
#__inference_bn2_layer_call_fn_14785�����Q�N
G�D
:�7
inputs+���������������������������
p 

 
� ";�8
unknown+����������������������������
>__inference_bn3_layer_call_and_return_conditional_losses_14875�����Q�N
G�D
:�7
inputs+���������������������������
p

 
� "F�C
<�9
tensor_0+���������������������������
� �
>__inference_bn3_layer_call_and_return_conditional_losses_14893�����Q�N
G�D
:�7
inputs+���������������������������
p 

 
� "F�C
<�9
tensor_0+���������������������������
� �
#__inference_bn3_layer_call_fn_14844�����Q�N
G�D
:�7
inputs+���������������������������
p

 
� ";�8
unknown+����������������������������
#__inference_bn3_layer_call_fn_14857�����Q�N
G�D
:�7
inputs+���������������������������
p 

 
� ";�8
unknown+����������������������������
C__inference_maxpool1_layer_call_and_return_conditional_losses_14759�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
(__inference_maxpool1_layer_call_fn_14754�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
C__inference_maxpool2_layer_call_and_return_conditional_losses_14831�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
(__inference_maxpool2_layer_call_fn_14826�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
C__inference_maxpool3_layer_call_and_return_conditional_losses_14903�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
(__inference_maxpool3_layer_call_fn_14898�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
D__inference_quant_bn1_layer_call_and_return_conditional_losses_14043}����;�8
1�.
(�%
inputs���������  
p
� "4�1
*�'
tensor_0���������  
� �
D__inference_quant_bn1_layer_call_and_return_conditional_losses_14061}����;�8
1�.
(�%
inputs���������  
p 
� "4�1
*�'
tensor_0���������  
� �
)__inference_quant_bn1_layer_call_fn_14012r����;�8
1�.
(�%
inputs���������  
p
� ")�&
unknown���������  �
)__inference_quant_bn1_layer_call_fn_14025r����;�8
1�.
(�%
inputs���������  
p 
� ")�&
unknown���������  �
D__inference_quant_bn2_layer_call_and_return_conditional_losses_14247}����;�8
1�.
(�%
inputs���������
p
� "4�1
*�'
tensor_0���������
� �
D__inference_quant_bn2_layer_call_and_return_conditional_losses_14265}����;�8
1�.
(�%
inputs���������
p 
� "4�1
*�'
tensor_0���������
� �
)__inference_quant_bn2_layer_call_fn_14216r����;�8
1�.
(�%
inputs���������
p
� ")�&
unknown����������
)__inference_quant_bn2_layer_call_fn_14229r����;�8
1�.
(�%
inputs���������
p 
� ")�&
unknown����������
D__inference_quant_bn3_layer_call_and_return_conditional_losses_14451}����;�8
1�.
(�%
inputs���������
p
� "4�1
*�'
tensor_0���������
� �
D__inference_quant_bn3_layer_call_and_return_conditional_losses_14469}����;�8
1�.
(�%
inputs���������
p 
� "4�1
*�'
tensor_0���������
� �
)__inference_quant_bn3_layer_call_fn_14420r����;�8
1�.
(�%
inputs���������
p
� ")�&
unknown����������
)__inference_quant_bn3_layer_call_fn_14433r����;�8
1�.
(�%
inputs���������
p 
� ")�&
unknown����������
F__inference_quant_conv1_layer_call_and_return_conditional_losses_13984{�./�;�8
1�.
(�%
inputs���������  
p
� "4�1
*�'
tensor_0���������  
� �
F__inference_quant_conv1_layer_call_and_return_conditional_losses_13999{�./�;�8
1�.
(�%
inputs���������  
p 
� "4�1
*�'
tensor_0���������  
� �
+__inference_quant_conv1_layer_call_fn_13944p�./�;�8
1�.
(�%
inputs���������  
p
� ")�&
unknown���������  �
+__inference_quant_conv1_layer_call_fn_13957p�./�;�8
1�.
(�%
inputs���������  
p 
� ")�&
unknown���������  �
F__inference_quant_conv2_layer_call_and_return_conditional_losses_14188{�_`�;�8
1�.
(�%
inputs���������
p
� "4�1
*�'
tensor_0���������
� �
F__inference_quant_conv2_layer_call_and_return_conditional_losses_14203{�_`�;�8
1�.
(�%
inputs���������
p 
� "4�1
*�'
tensor_0���������
� �
+__inference_quant_conv2_layer_call_fn_14148p�_`�;�8
1�.
(�%
inputs���������
p
� ")�&
unknown����������
+__inference_quant_conv2_layer_call_fn_14161p�_`�;�8
1�.
(�%
inputs���������
p 
� ")�&
unknown����������
F__inference_quant_conv3_layer_call_and_return_conditional_losses_14392}����;�8
1�.
(�%
inputs���������
p
� "4�1
*�'
tensor_0���������
� �
F__inference_quant_conv3_layer_call_and_return_conditional_losses_14407}����;�8
1�.
(�%
inputs���������
p 
� "4�1
*�'
tensor_0���������
� �
+__inference_quant_conv3_layer_call_fn_14352r����;�8
1�.
(�%
inputs���������
p
� ")�&
unknown����������
+__inference_quant_conv3_layer_call_fn_14365r����;�8
1�.
(�%
inputs���������
p 
� ")�&
unknown����������
G__inference_quant_dense1_layer_call_and_return_conditional_losses_14647r������4�1
*�'
!�
inputs����������
p
� ",�)
"�
tensor_0���������

� �
G__inference_quant_dense1_layer_call_and_return_conditional_losses_14667r������4�1
*�'
!�
inputs����������
p 
� ",�)
"�
tensor_0���������

� �
,__inference_quant_dense1_layer_call_fn_14582g������4�1
*�'
!�
inputs����������
p
� "!�
unknown���������
�
,__inference_quant_dense1_layer_call_fn_14599g������4�1
*�'
!�
inputs����������
p 
� "!�
unknown���������
�
H__inference_quant_flatten_layer_call_and_return_conditional_losses_14559l;�8
1�.
(�%
inputs���������
p
� "-�*
#� 
tensor_0����������
� �
H__inference_quant_flatten_layer_call_and_return_conditional_losses_14565l;�8
1�.
(�%
inputs���������
p 
� "-�*
#� 
tensor_0����������
� �
-__inference_quant_flatten_layer_call_fn_14548a;�8
1�.
(�%
inputs���������
p
� ""�
unknown�����������
-__inference_quant_flatten_layer_call_fn_14553a;�8
1�.
(�%
inputs���������
p 
� ""�
unknown�����������
I__inference_quant_maxpool1_layer_call_and_return_conditional_losses_14130s;�8
1�.
(�%
inputs���������  
p
� "4�1
*�'
tensor_0���������
� �
I__inference_quant_maxpool1_layer_call_and_return_conditional_losses_14135s;�8
1�.
(�%
inputs���������  
p 
� "4�1
*�'
tensor_0���������
� �
.__inference_quant_maxpool1_layer_call_fn_14120h;�8
1�.
(�%
inputs���������  
p
� ")�&
unknown����������
.__inference_quant_maxpool1_layer_call_fn_14125h;�8
1�.
(�%
inputs���������  
p 
� ")�&
unknown����������
I__inference_quant_maxpool2_layer_call_and_return_conditional_losses_14334s;�8
1�.
(�%
inputs���������
p
� "4�1
*�'
tensor_0���������
� �
I__inference_quant_maxpool2_layer_call_and_return_conditional_losses_14339s;�8
1�.
(�%
inputs���������
p 
� "4�1
*�'
tensor_0���������
� �
.__inference_quant_maxpool2_layer_call_fn_14324h;�8
1�.
(�%
inputs���������
p
� ")�&
unknown����������
.__inference_quant_maxpool2_layer_call_fn_14329h;�8
1�.
(�%
inputs���������
p 
� ")�&
unknown����������
I__inference_quant_maxpool3_layer_call_and_return_conditional_losses_14538s;�8
1�.
(�%
inputs���������
p
� "4�1
*�'
tensor_0���������
� �
I__inference_quant_maxpool3_layer_call_and_return_conditional_losses_14543s;�8
1�.
(�%
inputs���������
p 
� "4�1
*�'
tensor_0���������
� �
.__inference_quant_maxpool3_layer_call_fn_14528h;�8
1�.
(�%
inputs���������
p
� ")�&
unknown����������
.__inference_quant_maxpool3_layer_call_fn_14533h;�8
1�.
(�%
inputs���������
p 
� ")�&
unknown����������
F__inference_quant_relu1_layer_call_and_return_conditional_losses_14105wHI;�8
1�.
(�%
inputs���������  
p
� "4�1
*�'
tensor_0���������  
� �
F__inference_quant_relu1_layer_call_and_return_conditional_losses_14115wHI;�8
1�.
(�%
inputs���������  
p 
� "4�1
*�'
tensor_0���������  
� �
+__inference_quant_relu1_layer_call_fn_14070lHI;�8
1�.
(�%
inputs���������  
p
� ")�&
unknown���������  �
+__inference_quant_relu1_layer_call_fn_14079lHI;�8
1�.
(�%
inputs���������  
p 
� ")�&
unknown���������  �
F__inference_quant_relu2_layer_call_and_return_conditional_losses_14309wyz;�8
1�.
(�%
inputs���������
p
� "4�1
*�'
tensor_0���������
� �
F__inference_quant_relu2_layer_call_and_return_conditional_losses_14319wyz;�8
1�.
(�%
inputs���������
p 
� "4�1
*�'
tensor_0���������
� �
+__inference_quant_relu2_layer_call_fn_14274lyz;�8
1�.
(�%
inputs���������
p
� ")�&
unknown����������
+__inference_quant_relu2_layer_call_fn_14283lyz;�8
1�.
(�%
inputs���������
p 
� ")�&
unknown����������
F__inference_quant_relu3_layer_call_and_return_conditional_losses_14513y��;�8
1�.
(�%
inputs���������
p
� "4�1
*�'
tensor_0���������
� �
F__inference_quant_relu3_layer_call_and_return_conditional_losses_14523y��;�8
1�.
(�%
inputs���������
p 
� "4�1
*�'
tensor_0���������
� �
+__inference_quant_relu3_layer_call_fn_14478n��;�8
1�.
(�%
inputs���������
p
� ")�&
unknown����������
+__inference_quant_relu3_layer_call_fn_14487n��;�8
1�.
(�%
inputs���������
p 
� ")�&
unknown����������
H__inference_quant_softmax_layer_call_and_return_conditional_losses_14682c3�0
)�&
 �
inputs���������

p
� ",�)
"�
tensor_0���������

� �
H__inference_quant_softmax_layer_call_and_return_conditional_losses_14687c3�0
)�&
 �
inputs���������

p 
� ",�)
"�
tensor_0���������

� �
-__inference_quant_softmax_layer_call_fn_14672X3�0
)�&
 �
inputs���������

p
� "!�
unknown���������
�
-__inference_quant_softmax_layer_call_fn_14677X3�0
)�&
 �
inputs���������

p 
� "!�
unknown���������
�
I__inference_quantize_layer_layer_call_and_return_conditional_losses_13922w!";�8
1�.
(�%
inputs���������  
p
� "4�1
*�'
tensor_0���������  
� �
I__inference_quantize_layer_layer_call_and_return_conditional_losses_13931w!";�8
1�.
(�%
inputs���������  
p 
� "4�1
*�'
tensor_0���������  
� �
.__inference_quantize_layer_layer_call_fn_13892l!";�8
1�.
(�%
inputs���������  
p
� ")�&
unknown���������  �
.__inference_quantize_layer_layer_call_fn_13901l!";�8
1�.
(�%
inputs���������  
p 
� ")�&
unknown���������  �
#__inference_signature_wrapper_13349�B!"�./�����HI�_`�����yz����������������?�<
� 
5�2
0
input'�$
input���������  "=�:
8
quant_softmax'�$
quant_softmax���������
