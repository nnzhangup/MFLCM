function acc=test(data, model)

view_num = numel(data.X_test);
q=model.q;
A=model.A;
Z=model.Z;
P=model.P;
X_test=data.X_test;
Y_test=data.Y_test;
Ntest=length(Y_test);
Sum_AZ=zeros(q,Ntest);
for i=1:view_num
    Sum_AZ=Sum_AZ+A{i}*X_test{i};
end
Z=1/view_num*Sum_AZ;
Y_Predict=P*Z;
[~,ind]=max(Y_Predict);
 acc = sum(ind==Y_test)/length(Y_test)
end