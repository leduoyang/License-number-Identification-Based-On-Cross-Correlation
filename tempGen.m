%% tempGen-NUMBER
I=imread('tem_num.png');
I = rgb2gray(I);
I=im2double(I);
%%  binarization
I=binarization(I);
I=BoundaryExtension(I,5);
%% connected component labeling 
[L,N] = bwlabel(I);
for n=1:N
    [Y,X]=size(L);
    TEM=[];
    for y=1:Y
        for x=1:X
            if L(y,x)==n
                    TEM=[TEM;y x];
            end
        end
    end
    TEMPLATES{1,n}=TEM;
end
%% remove non-character components
COUNT=countObjects(TEMPLATES);
MEAN=mean(COUNT);
for i=1:length(COUNT)
    if COUNT(i)<(MEAN/10) 
        TEMPLATES{i}=[];
    end
end
c=1;
TEM={};
for i=1:length(TEMPLATES)
    if isempty(TEMPLATES{i})~=1
        TEM{c}=TEMPLATES{i};
        c=c+1;
    end
end
TEMPLATES=TEM;
%% extract objects with bounding box
TEMP_BOX=BoundingBOx(I,TEMPLATES);
LEN=length(TEMP_BOX);
c=1;
figure(3)
for l=1:LEN
    if isempty(TEMP_BOX{l})~=1
        subplot(6,6,c),imshow(TEMP_BOX{l});
        c=c+1;
    end
end
%% consider objects of bounding box as basic units to process
c=1;
OBJ={};
for i=1:length(TEMP_BOX)
    if isempty(TEMP_BOX{i})~=1
        OBJ{c}=TEMP_BOX{i};
        c=c+1;
    end
end
%% resize to 250*130 
%OBJ=resizeIMG(OBJ,60,45);
OBJ=ResizeIMG(OBJ,0.5);
for i=1:length(OBJ) %adjust the pixels to binary
    [Y,X]=size(OBJ{i});
    for y=1:Y
        for x=1:X
            if OBJ{i}(y,x)>0.5
                OBJ{i}(y,x)=1;
            else
                OBJ{i}(y,x)=0;
            end
        end
    end
end
TEMP=OBJ;
tem_num=TEMP;
save('tem_num','tem_num');
%% boundary
OBJ=TEMP;
TEMP_B={};
for N=1:length(OBJ)
    I=OBJ{N};[Y,X]=size(I);
    H=ones(3,3);
    [Hy,Hx]=size(H);
    c=1;
    n=length(H)-1; % # of lines for boundary extension
    for i=1:Hy
        for j=1:Hx
            G=Translation(i-2,j-2,I,n);
            if c~=1
                F=joint(F,G);
            else
                F=G;
            end
            c=c+1;
        end
    end
    F=F(n+1:n+Y,n+1:n+X);
    G=beta(I,F);
    TEMP_B{N}=G;
end
%% 
 [MEAN,N]=meanPerimeter(TEMP_B);
 num_P=floor(MEAN/5);
 RepPixels=cell(1,length(TEMP_B));
 for i=1:length(TEMP_B)
     TEM=TEMP_B{i};
     [Y,X]=size(TEM);
     [num,CC]=CCL(TEM);
     RP=[];
     for n=1:num
         c=CC{n};
         [len,~]=size(c);
         tem=zeros(Y,X);
         for l=1:len
                tem(c(l,1),c(l,2))=1;
         end
            [Center,~]=closestCentroid(tem,len);
            rp=setRepPoints(tem,Center,num_P,N{i});
            RP=[RP;rp];
     end
    RepPixels{i}=RP;
    TEM=0.25*TEM;
     [Y,~] = size(RepPixels{i});
     for j=1:Y
         TEM(RepPixels{i}(j,1),RepPixels{i}(j,2))=1;
     end
     TEMP_B{i}=TEM;
 end
 tem_num_B=TEMP_B;
save('tem_num_B','tem_num_B');
%%
LTS=length(tem_num_B);
c=1;
figure(3)
for l=1:LTS
    if isempty(tem_num_B{l})~=1
            subplot(6,6,c),imshow(tem_num_B{l});
            c=c+1;
    end
end
%% number of holes
NHTS=numOfH(TEMP);
 tem_num_NH=NHTS;
save('tem_num_NH','tem_num_NH');

close all
clear all
%% tempGen-NUMBER
I=imread('tem_char.png');
I = rgb2gray(I);
I=im2double(I);
%%  binarization
I=binarization(I);
I=BoundaryExtension(I,5);
%% connected component labeling 
[L,N] = bwlabel(I);
for n=1:N
    [Y,X]=size(L);
    TEM=[];
    for y=1:Y
        for x=1:X
            if L(y,x)==n
                    TEM=[TEM;y x];
            end
        end
    end
    TEMPLATES{1,n}=TEM;
end
%% remove non-character components
COUNT=countObjects(TEMPLATES);
MEAN=mean(COUNT);
for i=1:length(COUNT)
    if COUNT(i)<(MEAN/10) 
        TEMPLATES{i}=[];
    end
end
c=1;
TEM={};
for i=1:length(TEMPLATES)
    if isempty(TEMPLATES{i})~=1
        TEM{c}=TEMPLATES{i};
        c=c+1;
    end
end
TEMPLATES=TEM;
%% extract objects with bounding box
TEMP_BOX=BoundingBOx(I,TEMPLATES);
LEN=length(TEMP_BOX);
c=1;
figure(3)
for l=1:LEN
    if isempty(TEMP_BOX{l})~=1
        subplot(6,6,c),imshow(TEMP_BOX{l});
        c=c+1;
    end
end
%% consider objects of bounding box as basic units to process
c=1;
OBJ={};
for i=1:length(TEMP_BOX)
    if isempty(TEMP_BOX{i})~=1
        OBJ{c}=TEMP_BOX{i};
        c=c+1;
    end
end
%% resize to 250*130 
%OBJ=resizeIMG(OBJ,60,45);
OBJ=ResizeIMG(OBJ,0.5);
for i=1:length(OBJ) %adjust the pixels to binary
    [Y,X]=size(OBJ{i});
    for y=1:Y
        for x=1:X
            if OBJ{i}(y,x)>0.5
                OBJ{i}(y,x)=1;
            else
                OBJ{i}(y,x)=0;
            end
        end
    end
end
TEMP=OBJ;
tem_char=TEMP;
save('tem_char','tem_char');
%% boundary
OBJ=TEMP;
TEMP_B={};
for N=1:length(OBJ)
    I=OBJ{N};[Y,X]=size(I);
    H=ones(3,3);
    [Hy,Hx]=size(H);
    c=1;
    n=length(H)-1; % # of lines for boundary extension
    for i=1:Hy
        for j=1:Hx
            G=Translation(i-2,j-2,I,n);
            if c~=1
                F=joint(F,G);
            else
                F=G;
            end
            c=c+1;
        end
    end
    F=F(n+1:n+Y,n+1:n+X);
    G=beta(I,F);
    TEMP_B{N}=G;
end
%% 
 [MEAN,N]=meanPerimeter(TEMP_B);
 num_P=floor(MEAN/5);
 RepPixels=cell(1,length(TEMP_B));
 for i=1:length(TEMP_B)
     TEM=TEMP_B{i};
     [Y,X]=size(TEM);
     [num,CC]=CCL(TEM);
     RP=[];
     for n=1:num
         c=CC{n};
         [len,~]=size(c);
         tem=zeros(Y,X);
         for l=1:len
                tem(c(l,1),c(l,2))=1;
         end
            [Center,~]=closestCentroid(tem,len);
            rp=setRepPoints(tem,Center,num_P,N{i});
            RP=[RP;rp];
     end
    RepPixels{i}=RP;
    TEM=0.25*TEM;
     [Y,~] = size(RepPixels{i});
     for j=1:Y
         TEM(RepPixels{i}(j,1),RepPixels{i}(j,2))=1;
     end
     TEMP_B{i}=TEM;
 end
 tem_char_B=TEMP_B;
save('tem_char_B','tem_char_B');
%%
LTS=length(tem_char_B);
c=1;
figure(3)
for l=1:LTS
    if isempty(tem_char_B{l})~=1
            subplot(6,6,c),imshow(tem_char_B{l});
            c=c+1;
    end
end
%% number of holes
NHTS=numOfH(TEMP);
tem_char_NH=NHTS;
save('tem_char_NH','tem_char_NH');

close all
%% function
function img=BoundaryExtension(I,num)
    [Y,X]=size(I);
    img=zeros(Y+(num*2),X+(num*2));
    img(num+1:num+Y,num+1:num+X)=I;
    for n=num:-1:1
%         img(n,n)=img(n+1,n+1);
%         img(n,n+1:n+X)=img(n+1,(n+1):n+X);
%         img(n+1:n+Y,n)=img((n+1):n+Y,n+1);
        img(n,n)=0;
        img(n,n+1:n+X)=0;
        img(n+1:n+Y,n)=0;
        X=X+1;
        Y=Y+1;
    end
    [Y,X]=size(img);
%     img(1:Y,X-num+1:X)=repmat(img(1:Y,X-num),1,num);
%     img(Y-num+1:Y,1:X)=repmat(img(Y-num,1:X),num,1);
    img(1:Y,X-num+1:X)=0;
    img(Y-num+1:Y,1:X)=0;
end
function G=binarization(F)
    [Y,X]=size(F);
    [m,~]=min(min(F));
    s=std(F);
    for i=1:Y
        for j=1:X
            if F(i,j)>(m+s)
                F(i,j)=0;
            else
                F(i,j)=1;
            end
        end
    end
    G=F;
end
function O=CCLabeling(i,j,I)
    [Y,X]=size(I);
    tem=zeros(Y,X);
    for y=i-1:i+1 %G0
       for x=j-1:j+1
            tem(y,x)=I(i,j);
        end
    end
    c=0;
    b=0;
    while(b<c||c==0)
        b=c;
        for y=1:Y
            for x=1:X
                    if  I(y,x)~=tem(y,x)
                        tem(y,x)=0;
                    else
                        if I(y,x)==1
                            B=1;
                            if c~=0
                                [NN,~]=size(O);
                                if NN~=0
                                    for nn=1:NN
                                        if O(nn,1)==y && O(nn,2)==x
                                            B=0;
                                        end
                                    end
                                end
                            end
                            if B==1
                                c=c+1;
                                O(c,1)=y;
                                O(c,2)=x;
                            end
                        end
                    end
            end
        end
        H=ones(3,3);
        [Hy,Hx]=size(H);
        n=length(H)-1; % # of lines for boundary extension
        p=1;
        for i=1:Hy
            for j=1:Hx
                G=Translation(i-2,j-2,tem,n);
                if p~=1
                    F=union(F,G);
                else
                    F=G;
                end
                p=p+1;
             end
        end
        tem=F(n+1:n+Y,n+1:n+X);   
        
    end
end
function B=check(objects,i,j)
    N=length(objects);
    B=0;
    for n=1:N
        [X,~]=size(objects{n});
            for x=1:X
                if objects{n}(x,1)==i && objects{n}(x,2)==j
                    B=1;
                end
            end
    end
end
function F=joint(A,B)
    [Y,X]=size(A);
    F=zeros(Y,X);
    for i=1:Y
        for j=1:X
            if  A(i,j)==B(i,j)
                F(i,j)=A(i,j);
            end
        end
    end
end
function F=union(A,B)
    [Y,X]=size(A);
    F=zeros(Y,X);
    for i=1:Y
        for j=1:X
            if  A(i,j)==1 || B(i,j)==1
                F(i,j)=1;
            end
        end
    end
end
function G=beta(I,F)
    [Y,X]=size(I);
    G=zeros(Y,X);
    for i=1:Y
        for j=1:X
                G(i,j)=I(i,j)-F(i,j);
        end
    end
end
function T =Translation(y,x,I,n)
    [Y,X]=size(I);
    tem=zeros(Y,X);
    T=BoundaryExtension(tem,n);
    for i=1:Y
            for j=1:X
                T(n+i+y,n+j+x)=I(i,j);
            end
    end
end
function N=countObjects(OBJ)
    L=length(OBJ);
    N=zeros(1,L);
    for l=1:L
        object=OBJ{l};
        [Y,~]=size(object);
        N(l)=Y;
    end
end
function B=BoundingBOx(S,O)
    L=length(O);
    B={};
    for l=1:L
        tem=O{l};
        if isempty(tem)~=1
            [SY,SX]=size(S);
            BB=zeros(SY,SX);
            for i=1:length(tem)
                BB(tem(i,1),tem(i,2))=1;
            end
            Y=tem(:,1);
            X=tem(:,2);
            [Mx,~]=max(X);
            [mx,~]=min(X);
            [My,~]=max(Y);
             [my,~]=min(Y);
%              [len,s]=max([(My-my+1) (Mx-mx+1)]);
             I=S(my:My,mx:Mx);
%              shift_y=1;
%              shift_x=1;
%             if s ==2
%                 shift_y=floor(abs((My-my)-(Mx-mx))/2)+1;
%             else
%                 shift_x=floor(abs((My-my)-(Mx-mx))/2)+1;
%             end
%              I(shift_y:shift_y+(My-my),shift_x:shift_x+(Mx-mx))=S(my:My,mx:Mx);
        [Y,X]=size(I);
        TEM=zeros(Y+10,X+30);
        TEM(6:5+Y,16:15+X)=I;
         B{l}=TEM;
        end
    end
end
function [MEAN,N]=meanPerimeter(OBJ)
    c=0;
    C=0;
    for i=1:length(OBJ)
        tem=OBJ{i};
        [Y,X]=size(tem);
        for y=1:Y
            for x=1:X
                if tem(y,x)==1
                    c=c+1;
                    C=C+1;
                end
            end
        end
        N{i}=c;
        c=0;
    end
    MEAN=C/length(OBJ);
end
function [C,ONES]=closestCentroid(F,N)
    [Y,X]=size(F);    
    ONES=zeros(N,2);
    c=1;
    for y=1:Y
        for x=1:X
            if F(y,x)==1
                ONES(c,1)=y;
                ONES(c,2)=x;
                c=c+1;
            end
        end
    end
    center(1,1)=50;
    center(1,2)=50;
    m=compuD(center,ONES(1,:));index=1;
    for n=2:N
        dis=compuD(center,ONES(n,:));
        if dis<m
            m=dis;index=n;
        end
    end
    C=ONES(index,:);
end
function dis=compuD(A,B)
    dis=sqrt((A(1)-B(1))^2+(A(2)-B(2))^2);
end
function RP=setRepPoints(tem,Center,num_P,N)
    Gap=ceil(N/num_P);
    RP=Center;n=1;
    t=0;
    while(n<num_P)
        [L,~]=size(RP);
        for i=1:L
            t=t+1;
            c=RP(i,:);
            path=c;
            g=0;
            p=Recursive(tem,c,g,Gap,RP,path);
            if p~=0
                            B=0;
                             [len,~]=size(RP);
                            for l=1:len
                                rp=RP(l,:);
                                if abs(rp(1,1)-p(1,1))+abs(rp(1,2)-p(1,2))<Gap
                                    B=1;
                                end
                            end
                            t=0;
                            RP=[RP;p];
                            if B~=1
                                    n=n+1;
                            end
                            if n==num_P
                                break;
                            end
            end
        end
        if t>L
                return ;
        end
    end
end
function p=Recursive(tem,c,G,GAP,RP,path)
    p=0;g=G;
    for i=c(1,1)-1:c(1,1)+1
        for j=c(1,2)-1:c(1,2)+1
            if tem(i,j)==1
                B=0;
              [len,~]=size(RP);
              for l=1:len
                    if i==RP(l,1) && j==RP(l,2) 
                         B=1;
                     end
              end
              [len,~]=size(path);
              for l=1:len
                    if i==path(l,1) && j==path(l,2) 
                         B=1;
                     end
              end
               if B~=1
                        g=g+1;
                        if g==GAP
                                    p=[i,j];
                                    return;
                        else
                            path=[path;[i,j]];
                            p=Recursive(tem,[i,j],g,GAP,RP,path);
                            if p~=0
                                return;
                            else
                                path(end,:)=[];
                                g=G;
                            end
                        end
               end
            end
        end
    end
end
function [n,OBJ]=CCL(I)
        n=0; % # of objects
        OBJ = {};
        [Y,X]=size(I);
        for i=1:Y
            for j=1:X
                 if I(i,j)==1
                        if n~=0
                            B=check(OBJ,i,j);
                               if B==0
                                    n=n+1;
                                  OBJ{1,n}=CCLabeling(i,j,I);
                              end
                        else
                             n=n+1;
                            OBJ{1,n}=CCLabeling(i,j,I);
                        end
                end    
            end
        end
end
function N=numOfH(O)
    N=zeros(1,length(O));
    for i=1:length(O)
        o=O{i};
        [Y,X]=size(o);
        for y=1:Y
            for x=1:X
                o(y,x)=-(o(y,x)-1);
            end
        end
            n=0; % # of objects
            o=BoundaryExtension(o,11);
            [Y,X]=size(o);
            Object = {};
            for y=1:Y
                for x=1:X
                     if o(y,x)==1
                            if n~=0
                                B=check(Object,y,x);
                                   if B==0
                                        n=n+1;
                                      Object{1,n}=CCLabeling(y,x,o);
                                  end
                            else
                                 n=n+1;
                                Object{1,n}=CCLabeling(y,x,o);
                            end
                    end    
                end
            end
            COUNT=countObjects(Object);
            MEAN=mean(COUNT);
            for j=1:length(COUNT)
                if COUNT(j)<(MEAN/4) 
                    n=n-1;
                end
            end
            N(i)=n;
    end
end
function O=ResizeIMG(OBJ,R)
    O={};
    for i=1:length(OBJ)
    I=OBJ{i};
    I=imresize(I, R);
    O{i}=I;
    end
end







