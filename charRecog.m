%% load templates
load('tem_char');
load('tem_char_NH');
load('tem_num');
load('tem_num_NH');
%% unknown characters
I=imread('./test/1.jpg');
  I = rgb2gray(I);
 I=im2double(I);
%%  binarization
I = imbinarize(I);
I=-(I-1);
% I=imresize(I, 0.5);
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
    OBJ{1,n}=TEM;
end
%% remove non-character components
COUNT=countObjects(OBJ);
MEAN=mean(COUNT);
STD=std(COUNT);
for i=1:length(COUNT)
    if COUNT(i)<(STD+MEAN) 
        OBJ{i}=[];
    end
end
c=1;
TEM={};
for i=1:length(OBJ)
    if isempty(OBJ{i})~=1
        TEM{c}=OBJ{i};
        c=c+1;
    end
end
OBJ=TEM;
%load('OBJ');
%% sort by index
[NUM,CHAR]=divideNumChar(OBJ);
%% extract objects with bounding box
NUM_BOX=BoundingBOx(I,NUM);
CHAR_BOX=BoundingBOx(I,CHAR);
%% consider objects of bounding box as basic units to process
c=1;
NUM={};
for i=1:length(NUM_BOX)
    if isempty(NUM_BOX{i})~=1
        NUM{c}=NUM_BOX{i};
        c=c+1;
    end
end
c=1;
CHAR={};
for i=1:length(CHAR_BOX)
    if isempty(CHAR_BOX{i})~=1
        CHAR{c}=CHAR_BOX{i};
        c=c+1;
    end
end
%% resize to 200*200 for boundary extraction
% NUM=resizeIMG(NUM,60,30);
% for i=1:length(NUM) %adjust the pixels to binary
%     [Y,X]=size(NUM{i});
%     for y=1:Y
%         for x=1:X
%             if NUM{i}(y,x)>0.5
%                 NUM{i}(y,x)=1;
%             else
%                 NUM{i}(y,x)=0;
%             end
%         end
%     end
% end
% CHAR=resizeIMG(CHAR,60,48);
% for i=1:length(CHAR) %adjust the pixels to binary
%     [Y,X]=size(CHAR{i});
%     for y=1:Y
%         for x=1:X
%             if CHAR{i}(y,x)>0.5
%                 CHAR{i}(y,x)=1;
%             else
%                 CHAR{i}(y,x)=0;
%             end
%         end
%     end
% end
%% number of holes
% NHNUM=numOfH(NUM);
% NHCHAR=numOfH(CHAR);
%% recognition
figure(11)
RESULT2=recogXcorr(CHAR,tem_char);
% for i=1:length(CHAR)
%     char=NHCHAR(i);
%     for j=1:length(tem_char_NH)
%             if tem_char_NH(j)~=char
%                 RESULT2(i,j)=0;
%             end
%     end
% end
for i=1:length(CHAR)
    subplot(length(CHAR),4,4*(i-1)+1),imshow(CHAR{i});title('unknown character');
    [~,indx]=max(RESULT2(i,:));
    subplot(length(CHAR),4,4*(i-1)+2),imshow(tem_char{indx});title('character of recognition result');
    RESULT2(i,indx)=0;
    [~,indx]=max(RESULT2(i,:));
    subplot(length(CHAR),4,4*(i-1)+3),imshow(tem_char{indx});
    RESULT2(i,indx)=0;
end
figure(10)
RESULT1=recogXcorr(NUM,tem_num);
% for i=1:length(NUM)
%     num=NHNUM(i);
%     for j=1:length(tem_num_NH)
%             if tem_num_NH(j)~=num
%                 RESULT1(i,j)=0;
%             end
%     end
% end
for i=1:length(NUM)
    subplot(length(NUM),4,4*(i-1)+1),imshow(NUM{i});title('unknown character');
    [~,indx]=max(RESULT1(i,:));
    subplot(length(NUM),4,4*(i-1)+2),imshow(tem_num{indx});title('character of recognition result');
    RESULT1(i,indx)=0;
    [~,indx]=max(RESULT1(i,:));
    subplot(length(NUM),4,4*(i-1)+3),imshow(tem_num{indx});
    RESULT1(i,indx)=0;
end

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
function OBJ=ConnectIsolatedComp(OBJ,ISO)
   I=length(ISO);
    L=length(OBJ);
    indx=0;
    for i=1:I
        indx=0;
        isoC=OBJ{ISO(i)};
         for l=1:L
                if ISO(i)~=l
                    Comp=OBJ{l};
                    if isempty(isoC)~=1&&isempty(Comp)~=1
                    dis=distanceComputation(isoC,Comp);
                    end
                    if indx==0
                        indx=l;
                        m=dis;
                    else
                        if dis < m
                            m=dis;
                            indx=l;
                        end
                    end
                end
         end
         Comp=OBJ{indx};
         Comp=[Comp;isoC];
         OBJ{indx}=Comp;
         OBJ{ISO(i)}=[];
%             BB=zeros(512,512);
%             for ii=1:length(Comp)
%                 BB(Comp(ii,1),Comp(ii,2))=1;
%             end
%             figure(31)
%             imshow(BB);
%          for t=ISO(i)+1:L
%              OBJ{t-1}=OBJ{t};
%          end
%          L=length(OBJ);
    end
end
function D=distanceComputation(isoC,Comp)
    [N1,~]=size(isoC);
    [N2,~]=size(Comp);
    m=0;
    D=0;
    for n1=1:N1
        for n2=1:N2
            dis=(isoC(n1,1)-Comp(n2,1))^2+(isoC(n1,2)-Comp(n2,2))^2;
            D=D+dis;
            if n2==1
                    m=dis;
            else
                if dis < m
                    m=dis;
                end
            end
        end
    end
    %D=m;
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
        TEM=zeros(Y+20,X+40);
        TEM(11:10+Y,21:20+X)=I;
         B{l}=TEM;
        end
    end
end
function OBJ=resizeIMG(O,ROW,COL)
    for i=1:length(O)
        F=O{i};
        [in_rows,in_cols] = size(F);
         out_rows =ROW;
        out_cols = COL;
        S_R = in_rows / out_rows;
        S_C = in_cols / out_cols;
        [cf, rf] = meshgrid(1 : out_cols, 1 : out_rows);

        rf = rf * S_R;
        cf = cf * S_C;
        r = floor(rf);
        c = floor(cf);
        r(r < 1) = 1;
        c(c < 1) = 1;
        r(r > in_rows - 1) = in_rows - 1;
        c(c > in_cols - 1) = in_cols - 1;

        delta_R = rf - r;
        delta_C = cf - c;
        in1_ind = sub2ind([in_rows, in_cols], r, c);
        in2_ind = sub2ind([in_rows, in_cols], r+1,c);
        in3_ind = sub2ind([in_rows, in_cols], r, c+1);
        in4_ind = sub2ind([in_rows, in_cols], r+1, c+1);       

        G = zeros(out_rows, out_cols, size(F, 3));
        G = cast(G, class(F));

        for idx = 1 : size(F, 3)
            chan = double(F(:,:,idx)); 
            tmp = chan(in1_ind).*(1 - delta_R).*(1 - delta_C) + ...
                           chan(in2_ind).*(delta_R).*(1 - delta_C) + ...
                           chan(in3_ind).*(1 - delta_R).*(delta_C) + ...
                           chan(in4_ind).*(delta_R).*(delta_C);
            G(:,:,idx) = cast(tmp, class(F));
        end
        OBJ{i}=G;
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
function G=zeroDetection(F)
    [Y,X]=size(F);
    win_size=3;
    n=floor(win_size/2);
    tem_F=BoundaryExtension(F,win_size);
    for i=n+1:Y+n
        for j=n+1:X+n
                if tem_F(i,j)==0 
                        tem=sort(tem_F(i-n:i+n,j-n:j+n));
                        tem_F(i,j)=tem(round(win_size*win_size*3/4));
                end
        end
    end
    G=tem_F(n+1:n+Y,n+1:n+X);
end
function G=medianFilter(F)
    [Y,X]=size(F);
% remove noise of G2 : median filter
    win_size=3;
    n=floor(win_size/2);
    tem_F=BoundaryExtension(F,win_size);
    for i=n+1:Y+n
        for j=n+1:X+n
            tem=sort(tem_F(i-n:i+n,j-n:j+n));
            tem_F(i,j)=tem(round(win_size*win_size/2));
        end
    end
    G=tem_F(n+1:n+Y,n+1:n+X);
end
function RESULT=recogWdis(OBJ,TS)
    LEN=length(OBJ);
    [~,N]=size(TS);
    for l=1:LEN
            obj=OBJ{l};
            [Y,X]=size(obj);
            L=0;
            for y=1:Y
                    for x=1:X
                        if obj(y,x)==1
                            L=L+1;
                        end
                    end
            end
            result=[];
                for j=1:N
                    tem=TransferCost(obj,TS{l,j},L);
                    result=[result tem];
                end
            RESULT(l,:)=result;
    end
end
function cost=TransferCost(obj,template,L)
    cost=0;
    [Y,X]=size(template);
    COL=0;
    t_index=[];
    for y=1:Y
        for x=1:X
            if template(y,x)==1
                COL=COL+1;
                t_index(COL,1)=y;
                t_index(COL,2)=x;
            end
        end
    end
    [Y,X]=size(obj);
    ROW=0;
    o_index=[];    
    for y=1:Y
        for x=1:X
            if obj(y,x)==1
                ROW=ROW+1;
                o_index(ROW,1)=y;
                o_index(ROW,2)=x;
            end
        end
    end    
    DIS=zeros(ROW,COL);
    for c=1:COL
        for r=1:ROW
            DIS(r,c)=(o_index(r,1)-t_index(c,1))^2+(o_index(r,2)-t_index(c,2))^2;
        end
    end
    for l=1:L
        if isempty(DIS)~=1
        [M,II]=min(DIS);
        [M,I]=min(M);
        cost=cost+M;
        if isempty(DIS)~=1
            DIS(:,I)=1000000;
        end
         if isempty(DIS)~=1
                 DIS(II(I),:)=1000000;
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
function R=recogXcorr(O,T)
    R=zeros(length(O),length(T));
    for i=1:length(O)
        obj = O{i};
            for j=1:length(T)
                temp = T{j};
                [Y,X]=size(temp);
                obj=resizeI(obj,Y,X);
                c=1;
                [Y,X]=size(temp);
                for y=1:Y
                    for x=1:X
                        if temp(y,x)==1
                            c=c+1;
                        end
                    end
                 end
%                 imshow(obj);
%                 imshow(temp);
                R(i,j)=sum(sum(obj.*temp))/c;
            end
    end
end
function OBJ=resizeI(O,ROW,COL)
        F=O;
        [in_rows,in_cols] = size(F);
         out_rows =ROW;
        out_cols = COL;
        S_R = in_rows / out_rows;
        S_C = in_cols / out_cols;
        [cf, rf] = meshgrid(1 : out_cols, 1 : out_rows);

        rf = rf * S_R;
        cf = cf * S_C;
        r = floor(rf);
        c = floor(cf);
        r(r < 1) = 1;
        c(c < 1) = 1;
        r(r > in_rows - 1) = in_rows - 1;
        c(c > in_cols - 1) = in_cols - 1;

        delta_R = rf - r;
        delta_C = cf - c;
        in1_ind = sub2ind([in_rows, in_cols], r, c);
        in2_ind = sub2ind([in_rows, in_cols], r+1,c);
        in3_ind = sub2ind([in_rows, in_cols], r, c+1);
        in4_ind = sub2ind([in_rows, in_cols], r+1, c+1);       

        G = zeros(out_rows, out_cols, size(F, 3));
        G = cast(G, class(F));

        for idx = 1 : size(F, 3)
            chan = double(F(:,:,idx)); 
            tmp = chan(in1_ind).*(1 - delta_R).*(1 - delta_C) + ...
                           chan(in2_ind).*(delta_R).*(1 - delta_C) + ...
                           chan(in3_ind).*(1 - delta_R).*(delta_C) + ...
                           chan(in4_ind).*(delta_R).*(delta_C);
            G(:,:,idx) = cast(tmp, class(F));
        end
        OBJ=G;
end
function [num,char]=divideNumChar(OBJ)
    num={};char={};
    index=[];
    for i=1:length(OBJ)
        pixels=OBJ{i};
        L=length(pixels);
        min_y=pixels(1,1);
        min_x=pixels(1,2);
        for l=1:L
                    if pixels(l,1)<min_y && pixels(l,2)<min_x
                        min_y=pixels(l,1);min_x=pixels(l,2);
                    end
        end
        index=[index min_x];
    end
    for i=1:length(index)
        [~,I]=min(index);
        if i<4
            char{i}=OBJ{I};
        else
            num{i-3}=OBJ{I};
        end
         index(I)=10000000;
    end
end






