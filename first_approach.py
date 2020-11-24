import pandas as pd
from networkx.readwrite import json_graph
import numpy as np

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.extmath import density
from sklearn import metrics

import math


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return plt


x_data=pd.read_json('./dataset/noduplicatedataset.json',lines=True)
x_data.head()
print(x_data)
feature_cols = ['id','semantic','lista_asm']
X_all =  x_data.loc[:,'lista_asm']
Y_all =  x_data.loc[:,'semantic']

X_train, X_test, y_train, y_test = train_test_split(X_all, Y_all, test_size=0.333, 
                                                    random_state=42)

vectorizer = HashingVectorizer()

X_train = vectorizer.transform(X_train)
X_test  = vectorizer.transform(X_test)
clf     = LinearSVC()
model   = clf.fit(X_train,y_train)
pred    = clf.predict(X_test)
score   = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
print("classification report:")
print(metrics.classification_report(y_test, pred,))
print("confusion matrix:")
print(metrics.confusion_matrix(y_test, pred))

plot_confusion_matrix(y_test,pred, classes=np.array(['encryption','math','sort','string']), normalize=False).show()

smsnew1 = np.array(["['jmp qword ptr [rip + 0x203b72]', 'jmp qword ptr [rip + 0x203b6a]', 'jmp qword ptr [rip + 0x203b62]', 'jmp qword ptr [rip + 0x203b52]', 'push rbp', 'push r15', 'push r14', 'push r13', 'push r12', 'push rbx', 'sub rsp, 0x18', 'mov r14, rdx', 'mov rbx, rsi', 'mov ebp, edi', 'mov edi, 0x403504', 'call 0xfffffffffffffe50', 'push rbp', 'push r15', 'push r14', 'push r13', 'push r12', 'push rbx', 'mov eax, edi', 'mov qword ptr [rsp - 0x28], rax', 'cmp dword ptr [rax*4 + 0x604074], 0', 'jle 0x1ca', 'push rbp', 'push r15', 'push r14', 'push r13', 'push r12', 'push rbx', 'sub rsp, 0x148', 'mov ebx, edx', 'mov qword ptr [rsp + 0x20], rsi', 'xorps xmm0, xmm0', 'movaps xmmword ptr [rsp + 0x130], xmm0', 'movaps xmmword ptr [rsp + 0x120], xmm0', 'movaps xmmword ptr [rsp + 0x110], xmm0', 'movaps xmmword ptr [rsp + 0x100], xmm0', 'movaps xmmword ptr [rsp + 0xf0], xmm0', 'movaps xmmword ptr [rsp + 0xe0], xmm0', 'movaps xmmword ptr [rsp + 0xd0], xmm0', 'movaps xmmword ptr [rsp + 0xc0], xmm0', 'movaps xmmword ptr [rsp + 0xb0], xmm0', 'movaps xmmword ptr [rsp + 0xa0], xmm0', 'movaps xmmword ptr [rsp + 0x90], xmm0', 'movaps xmmword ptr [rsp + 0x80], xmm0', 'movaps xmmword ptr [rsp + 0x70], xmm0', 'movaps xmmword ptr [rsp + 0x60], xmm0', 'movaps xmmword ptr [rsp + 0x50], xmm0', 'movaps xmmword ptr [rsp], xmm0', 'lea rdx, [rsp + 0x50]', 'mov dword ptr [rsp + 0x1c], edi', 'mov rsi, rcx', 'call 0xffffffffffffe9a0', 'mov dword ptr [rsp + 0x28], ebx', 'test ebx, ebx', 'jle 0x55b', 'mov r15d, dword ptr [rsp + 0x1c]', 'lea r13, [rsp]', 'xor ecx, ecx', 'mov qword ptr [rsp + 0x40], r15', 'nop dword ptr [rax + rax]', 'mov edi, 0x4034d8', 'xor eax, eax', 'mov rbx, rcx', 'mov esi, ebx', 'call 0xffffffffffffe755', 'mov edi, 0x4034d8', 'xor eax, eax', 'mov rbx, rcx', 'mov esi, ebx', 'call 0xffffffffffffe740', 'mov ebp, dword ptr [r15*4 + 0x604074]', 'test ebp, ebp', 'jle 0x32', 'lea edx, [rbp*4]', 'movsxd rsi, ebx', 'add rsi, qword ptr [rsp + 0x20]', 'test edx, edx', 'mov eax, 1', 'cmovle edx, eax', 'dec edx', 'inc rdx', 'mov rdi, r13', 'call 0xffffffffffffe743', 'mov qword ptr [rsp + 0x30], rbx', 'mov r12d, dword ptr [r15*4 + 0x60405c]', 'test r12d, r12d', 'js 0x44f', 'lea r14d, [r12*4]', 'nop word ptr cs:[rax + rax]', 'mov edi, 0x40349b', 'xor eax, eax', 'mov esi, r12d', 'call 0xffffffffffffe6e7', 'mov edi, 0x40349b', 'xor eax, eax', 'mov esi, r12d', 'call 0xffffffffffffe6d0', 'mov ebx, dword ptr [r15*4 + 0x604074]', 'mov edi, 0x403470', 'mov esi, 0x4034a7', 'xor eax, eax', 'call 0xffffffffffffe6c1', 'test ebx, ebx', 'jle 0x2f', 'shl ebx, 2', 'mov rbp, r13', 'nop word ptr cs:[rax + rax]', 'movzx esi, byte ptr [rbp]', 'mov edi, 0x403478', 'xor eax, eax', 'call 0xffffffffffffe6a4', 'movzx esi, byte ptr [rbp]', 'mov edi, 0x403478', 'xor eax, eax', 'call 0xffffffffffffe690', 'inc rbp', 'dec ebx', 'jne 0xfffffffffffffff0', 'mov edi, 0xa', 'call 0xffffffffffffe659', 'movsxd rbx, dword ptr [r15*4 + 0x604074]', 'test rbx, rbx', 'jle 0x6a', 'mov eax, ebx', 'and eax, 1', 'cmp ebx, 1', 'mov ecx, 0', 'je 0x46', 'mov rdx, rbx', 'sub rdx, rax', 'mov esi, ebx', 'imul esi, r12d', 'xor ecx, ecx', 'nop dword ptr [rax + rax]', 'lea rdi, [rsi + rcx]', 'movsxd rdi, edi', 'mov ebp, dword ptr [rsp + rdi*4 + 0x50]', 'xor dword ptr [rsp + rcx*4], ebp', 'inc edi', 'movsxd rdi, edi', 'mov edi, dword ptr [rsp + rdi*4 + 0x50]', 'xor dword ptr [rsp + rcx*4 + 4], edi', 'add rcx, 2', 'cmp rdx, rcx', 'jne 0x13', 'lea rdi, [rsi + rcx]', 'movsxd rdi, edi', 'mov ebp, dword ptr [rsp + rdi*4 + 0x50]', 'xor dword ptr [rsp + rcx*4], ebp', 'inc edi', 'movsxd rdi, edi', 'mov edi, dword ptr [rsp + rdi*4 + 0x50]', 'xor dword ptr [rsp + rcx*4 + 4], edi', 'add rcx, 2', 'cmp rdx, rcx', 'jne 0', 'test rax, rax', 'je 0x17', 'mov eax, ebx', 'imul eax, r12d', 'lea eax, [rcx + rax]', 'cdqe ', 'mov eax, dword ptr [rsp + rax*4 + 0x50]', 'xor dword ptr [rsp + rcx*4], eax', 'mov edi, 0x403470', 'mov esi, 0x4034c1', 'xor eax, eax', 'call 0xffffffffffffe617', 'mov edi, 0x403470', 'mov esi, 0x4034c1', 'xor eax, eax', 'call 0xffffffffffffe605', 'test ebx, ebx', 'jle 0x3a', 'lea ebp, [rbx*4]', 'imul ebx, r14d', 'movsxd rax, ebx', 'lea rbx, [rsp + rax + 0x50]', 'nop word ptr cs:[rax + rax]', 'movzx esi, byte ptr [rbx]', 'mov edi, 0x403478', 'xor eax, eax', 'call 0xffffffffffffe5f0', 'movzx esi, byte ptr [rbx]', 'mov edi, 0x403478', 'xor eax, eax', 'call 0xffffffffffffe5d0', 'inc rbx', 'dec ebp', 'jne 0xfffffffffffffff1', 'mov edi, 0xa', 'call 0xffffffffffffe59a', 'test r12d, r12d', 'mov dword ptr [rsp + 0x2c], r14d', 'je 0x270', 'cmp r12d, dword ptr [r15*4 + 0x60405c]', 'jge 0x5f', 'mov ebx, dword ptr [r15*4 + 0x604074]', 'mov edi, 0x403470', 'mov esi, 0x4034bb', 'xor eax, eax', 'call 0xffffffffffffe598', 'test ebx, ebx', 'jle 0x26', 'shl ebx, 2', 'mov rbp, r13', 'nop dword ptr [rax + rax]', 'movzx esi, byte ptr [rbp]', 'mov edi, 0x403478', 'xor eax, eax', 'call 0xffffffffffffe57b', 'movzx esi, byte ptr [rbp]', 'mov edi, 0x403478', 'xor eax, eax', 'call 0xffffffffffffe570', 'inc rbp', 'dec ebx', 'jne 0xfffffffffffffff0', 'mov edi, 0xa', 'call 0xffffffffffffe539', 'mov edi, dword ptr [rsp + 0x1c]', 'mov rsi, r13', 'call 0xfffffffffffffb6f', 'mov qword ptr [rsp + 0x38], r12', 'mov ebx, dword ptr [r15*4 + 0x604074]', 'mov edi, 0x403470', 'mov esi, 0x4034b3', 'xor eax, eax', 'call 0xffffffffffffe543', 'test ebx, ebx', 'jle 0x2c', 'shl ebx, 2', 'mov rbp, r13', 'nop word ptr cs:[rax + rax]', 'movzx esi, byte ptr [rbp]', 'mov edi, 0x403478', 'xor eax, eax', 'call 0xffffffffffffe521', 'movzx esi, byte ptr [rbp]', 'mov edi, 0x403478', 'xor eax, eax', 'call 0xffffffffffffe510', 'inc rbp', 'dec ebx', 'jne 0xfffffffffffffff0', 'mov edi, 0xa', 'call 0xffffffffffffe4d9', 'movsxd r12, dword ptr [r15*4 + 0x604074]', 'cmp r12, 2', 'jl 0xdd', 'lea eax, [r12*4]', 'mov qword ptr [rsp + 0x48], rax', 'mov r10d, r12d', 'and r10d, 1', 'mov rdx, r12', 'sub rdx, r10', 'mov r9d, 5', 'mov edi, 1', 'nop dword ptr [rax + rax]', 'cmp r12, rdi', 'jle 0xbc', 'cmp r12, rdi', 'jle 0x8f', 'mov rax, qword ptr [rsp + 0x48]', 'lea eax, [rax + rdi - 4]', 'movsxd r11, eax', 'mov r13d, r12d', 'sub r13d, edi', 'xor eax, eax', 'nop dword ptr [rax]', 'cmp r12d, 1', 'mov r8b, byte ptr [rsp + rdi]', 'mov esi, 0', 'je 0x63', 'cmp r12d, 1', 'mov r8b, byte ptr [rsp + rdi]', 'mov esi, 0', 'je 0x4c', 'mov ebx, r9d', 'xor esi, esi', 'nop word ptr cs:[rax + rax]', 'movsxd rbx, ebx', 'movzx r14d, byte ptr [rsp + rbx]', 'lea r15d, [rbx - 4]', 'movsxd rbp, r15d', 'mov byte ptr [rsp + rbp], r14b', 'add rsi, 2', 'lea ebp, [rbx + 4]', 'movsxd rbp, ebp', 'movzx ecx, byte ptr [rsp + rbp]', 'mov byte ptr [rsp + rbx], cl', 'lea ebx, [rbx + 8]', 'cmp rdx, rsi', 'jne 0x11', 'movsxd rbx, ebx', 'movzx r14d, byte ptr [rsp + rbx]', 'lea r15d, [rbx - 4]', 'movsxd rbp, r15d', 'mov byte ptr [rsp + rbp], r14b', 'add rsi, 2', 'lea ebp, [rbx + 4]', 'movsxd rbp, ebp', 'movzx ecx, byte ptr [rsp + rbp]', 'mov byte ptr [rsp + rbx], cl', 'lea ebx, [rbx + 8]', 'cmp rdx, rsi', 'jne 0', 'test r10, r10', 'je 0x18', 'lea ecx, [rdi + rsi*4 + 4]', 'movsxd rcx, ecx', 'mov cl, byte ptr [rsp + rcx]', 'lea esi, [rdi + rsi*4]', 'movsxd rsi, esi', 'mov byte ptr [rsp + rsi], cl', 'mov byte ptr [rsp + r11], r8b', 'inc eax', 'cmp eax, r13d', 'jl 0xffffffffffffffaf', 'mov byte ptr [rsp + r11], r8b', 'inc eax', 'cmp eax, r13d', 'jl 0xffffffffffffff9c', 'inc rdi', 'inc r9d', 'cmp rdi, r12', 'jl 0xffffffffffffff71', 'mov edi, 0x403470', 'mov esi, 0x4034ad', 'xor eax, eax', 'call 0xffffffffffffe412', 'test r12d, r12d', 'lea r13, [rsp]', 'jle 0x28', 'shl r12d, 2', 'mov rbx, r13', 'nop ', 'movzx esi, byte ptr [rbx]', 'mov edi, 0x403478', 'xor eax, eax', 'call 0xffffffffffffe3f8', 'movzx esi, byte ptr [rbx]', 'mov edi, 0x403478', 'xor eax, eax', 'call 0xffffffffffffe3f0', 'inc rbx', 'dec r12d', 'jne 0xfffffffffffffff1', 'mov edi, 0xa', 'call 0xffffffffffffe3b9', 'mov r15, qword ptr [rsp + 0x40]', 'movsxd r14, dword ptr [r15*4 + 0x604074]', 'test r14, r14', 'jle 0xdf', 'xor eax, eax', 'xor ecx, ecx', 'nop dword ptr [rax + rax]', 'cdqe ', 'movzx edx, byte ptr [rsp + rax]', 'movzx edx, byte ptr [rdx + 0x403310]', 'mov byte ptr [rsp + rax], dl', 'mov rdx, rax', 'or rdx, 1', 'movzx esi, byte ptr [rsp + rdx]', 'movzx ebx, byte ptr [rsi + 0x403310]', 'mov byte ptr [rsp + rdx], bl', 'mov rdx, rax', 'or rdx, 2', 'movzx esi, byte ptr [rsp + rdx]', 'movzx ebx, byte ptr [rsi + 0x403310]', 'mov byte ptr [rsp + rdx], bl', 'mov rdx, rax', 'or rdx, 3', 'movzx esi, byte ptr [rsp + rdx]', 'movzx ebx, byte ptr [rsi + 0x403310]', 'mov byte ptr [rsp + rdx], bl', 'inc rcx', 'add eax, 4', 'cmp rcx, r14', 'jl 9', 'cdqe ', 'movzx edx, byte ptr [rsp + rax]', 'movzx edx, byte ptr [rdx + 0x403310]', 'mov byte ptr [rsp + rax], dl', 'mov rdx, rax', 'or rdx, 1', 'movzx esi, byte ptr [rsp + rdx]', 'movzx ebx, byte ptr [rsi + 0x403310]', 'mov byte ptr [rsp + rdx], bl', 'mov rdx, rax', 'or rdx, 2', 'movzx esi, byte ptr [rsp + rdx]', 'movzx ebx, byte ptr [rsi + 0x403310]', 'mov byte ptr [rsp + rdx], bl', 'mov rdx, rax', 'or rdx, 3', 'movzx esi, byte ptr [rsp + rdx]', 'movzx ebx, byte ptr [rsi + 0x403310]', 'mov byte ptr [rsp + rdx], bl', 'inc rcx', 'add eax, 4', 'cmp rcx, r14', 'jl 0', 'mov r12, qword ptr [rsp + 0x38]', 'jmp 0x1e', 'mov r14d, dword ptr [r15*4 + 0x604074]', 'mov edi, 0x403470', 'mov esi, 0x4034c9', 'xor eax, eax', 'call 0xffffffffffffe340', 'mov edi, 0x403470', 'mov esi, 0x4034c9', 'xor eax, eax', 'call 0xffffffffffffe338', 'test r14d, r14d', 'jle 0x4d', 'shl r14d, 2', 'mov rbp, r13', 'nop word ptr cs:[rax + rax]', 'movzx esi, byte ptr [rbp]', 'mov edi, 0x403478', 'xor eax, eax', 'call 0xffffffffffffe322', 'movzx esi, byte ptr [rbp]', 'mov edi, 0x403478', 'xor eax, eax', 'call 0xffffffffffffe310', 'inc rbp', 'dec r14d', 'jne 0xfffffffffffffff0', 'jmp 0x1e', 'mov edi, 0x403470', 'mov esi, 0x4034c9', 'xor eax, eax', 'call 0xffffffffffffe2f0', 'mov r12, qword ptr [rsp + 0x38]', 'mov edi, 0xa', 'call 0xffffffffffffe2bf', 'mov edi, 0xa', 'call 0xffffffffffffe2ba', 'mov r14d, dword ptr [rsp + 0x2c]', 'add r14d, -4', 'test r12d, r12d', 'lea eax, [r12 - 1]', 'mov r12d, eax', 'jg 0xfffffffffffffc00', 'mov ebp, dword ptr [r15*4 + 0x604074]', 'mov rax, qword ptr [rsp + 0x30]', 'movsxd rbx, eax', 'test ebp, ebp', 'jle 0x41', 'mov rax, qword ptr [rsp + 0x30]', 'movsxd rbx, eax', 'test ebp, ebp', 'jle 0x39', 'mov rax, qword ptr [rsp + 0x20]', 'lea rax, [rax + rbx]', 'xor ecx, ecx', 'nop dword ptr [rax]', 'movzx edx, byte ptr [rsp + rcx]', 'mov byte ptr [rax + rcx], dl', 'inc rcx', 'movsxd rdx, dword ptr [r15*4 + 0x604074]', 'shl rdx, 2', 'cmp rcx, rdx', 'jl 0x12', 'movzx edx, byte ptr [rsp + rcx]', 'mov byte ptr [rax + rcx], dl', 'inc rcx', 'movsxd rdx, dword ptr [r15*4 + 0x604074]', 'shl rdx, 2', 'cmp rcx, rdx', 'jl 0', 'mov edi, 0x403513', 'call 0xffffffffffffe265', 'mov ebp, dword ptr [r15*4 + 0x604074]', 'mov edi, 0x403470', 'mov esi, 0x4034f4', 'xor eax, eax', 'call 0xffffffffffffe26b', 'test ebp, ebp', 'jle 0x28', 'add rbx, qword ptr [rsp + 0x20]', 'shl ebp, 2', 'nop word ptr [rax + rax]', 'movzx esi, byte ptr [rbx]', 'mov edi, 0x403478', 'xor eax, eax', 'call 0xffffffffffffe24e', 'movzx esi, byte ptr [rbx]', 'mov edi, 0x403478', 'xor eax, eax', 'call 0xffffffffffffe240', 'inc rbx', 'dec ebp', 'jne 0xfffffffffffffff1', 'mov edi, 0xa', 'call 0xffffffffffffe20a', 'mov eax, dword ptr [r15*4 + 0x604074]', 'mov rcx, qword ptr [rsp + 0x30]', 'lea ecx, [rcx + rax*4]', 'cmp ecx, dword ptr [rsp + 0x28]', 'jl 0xfffffffffffffae0', 'xor eax, eax', 'add rsp, 0x148', 'pop rbx', 'pop r12', 'pop r13', 'pop r14', 'pop r15', 'pop rbp', 'ret ']"])
xnew1   = vectorizer.transform(smsnew1)
ynew1 = model.predict(xnew1)
print('%s' %(ynew1))
def evaluate_mystery_set(clf):
    x_data=pd.read_json('./dataset/nodupblindtest.json',lines=True)
    x_data.head()
    vectorizer = HashingVectorizer()
    X_test = vectorizer.transform(x_data.lista_asm)
    y = clf.predict(X_test)
    return y
evaluate_mystery_set(clf)
