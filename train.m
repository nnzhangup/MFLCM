function model=train(data, model)

view_num = numel(data.X_train);
N=length(data.Y_train);
r=model.r;
Z=model.Z;
X_train=data.X_train;
A=model.A;
D=model.D;
q=model.q;
theta=model.theta;
S=model.S;
H=model.H; 
P=model.P;
W=model.W;
lamda=model.lamda;
%%update A
for v=1:view_num
    %A{v}=(D{v}'*D{v}+r*eye(q))\(D{v}'+r*Z*X_train{v}'/(X_train{v}*X_train{v}'));
    A{v}=(D{v}'*D{v}+r*eye(q))\(D{v}'+r*Z/X_train{v});
    %D{v}=X_train{v}*X_train{v}'*A{v}'/(A{v}*X_train{v}*X_train{v}'*A{v}');
    D{v}=eye(size(X_train{v},1))/A{v};
end
%P=q*S*Z'/(Z*Z')+(theta/lamda)*(H-Z)/(Z*Z');
P=(q*S*Z'+(theta/lamda)*W)/(Z*Z'+(theta/lamda)*eye(q));
W=sign(P);
H=sign(Z);
Sum_AZ=zeros(q,N);
for i=1:view_num
    Sum_AZ=Sum_AZ+A{i}*X_train{i};
end
Z=((r*view_num+theta)*eye(q)+lamda*P'*P)\(r*Sum_AZ+lamda*q*P'*S+theta*H);


L1=0;
for i=1:view_num
    L1=L1+(norm((X_train{i}-D{i}*A{i}*X_train{i}),'fro'))^2;
end
L2=0;
for i=1:view_num
    L2=L2+(norm((A{i}*X_train{i}-Z),'fro'))^2;
end
L2=r*L2;

L3=0;
L3=lamda*(norm((P*Z-q*S),'fro'))^2;

L4=0;
L4=theta*(norm((Z-H),'fro'))^2+theta*((norm((P-W),'fro'))^2);
model.loss=L1+L2+L3+L4;
model.loss=model.loss/N;

model.A=A;
model.D=D;
model.H=H;
model.P=P;
model.W=W;
model.Z=Z;
end