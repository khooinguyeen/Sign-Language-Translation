a=input("Nhập xâu S: ");
b=list(a);
b.reverse();
b=''.join(b);
if a==b:
    print("Xâu đối xứng!");
else:
    print("Xâu không đối xứng!");