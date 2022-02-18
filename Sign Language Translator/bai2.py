a = input("Nhập xâu: ")
kitu= ''
for i in a:
    if i.lower() not in kitu:
        kitu+=i
for i in kitu:
    count =0
    for j in a:
        if  i.lower()==j.lower():
            count+=1;
    print(f"Kí tự {i.lower()} xuất hiện {count} lần")