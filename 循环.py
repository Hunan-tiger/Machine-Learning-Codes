for i in range(1,10):
   for j in range(1,i+1):
    ans=i*j
    if i!=j:
        print(f"{i}*{j}={ans}",end=' ')
    else:
        print(f"{i}*{j}={ans}")#f表示后面的字符串里有外部变量,告诉解释器大括号里面的是一个变量
