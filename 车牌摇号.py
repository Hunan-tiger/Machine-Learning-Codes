print("您好，欢迎使用湖南长沙车牌摇号系统")
print("如果您已准备好，请输入'开始'")
write=input("请输入：")
print()
if write=="开始":
    count=0
    while count<3:
        import random
        import string
        s=string.ascii_uppercase+string.digits
        c=random.sample(s,5)
        b="".join(c)
        print("您的车牌号为湘A-"+f"{b}")
        print()
        print("您是否接受该号码？")
        print()
        ans=int(input("接受输入1，不接受输入0："))
        if ans==1:
            print("感谢您的使用，再见！")
            exit()
        else:
            count+=1
            if count==2:
                print("请注意，这是您的最后一次摇号机会!!!")
                continue
            else:
                print()
                continue
else:
    print("您的输入有误，请重新进入该系统。")
    exit()
        
    