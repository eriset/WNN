clear all
close all



%% SECTION 1 CONSTRUCTING FILES S_i of images of digit i (i=0,...,9) from training and test images
n=784;
d=load('mnist.mat');
Xtr=double(d.trainX);
Ytr=double(d.trainY);
Xte=double(d.testX);
Yte=double(d.testY);


S_0=zeros(1,784);
S_1=zeros(1,784);
S_2=zeros(1,784);
S_3=zeros(1,784);
S_4=zeros(1,784);
S_5=zeros(1,784);
S_6=zeros(1,784);
S_7=zeros(1,784);
S_8=zeros(1,784);
S_9=zeros(1,784);
M0=0;
M1=0;
M2=0;
M3=0;
M4=0;
M5=0;
M6=0;
M7=0;
M8=0;
M9=0;
for i=1:60000
    if Ytr(i)==0
        M0=M0+1;
        S_0(M0,:)=Xtr(i,:);
    end
    if Ytr(i)==1
        M1=M1+1;
        S_1(M1,:)=Xtr(i,:);
    end
    if Ytr(i)==2
        M2=M2+1;
        S_2(M2,:)=Xtr(i,:);
    end
    if Ytr(i)==3
        M3=M3+1;
        S_3(M3,:)=Xtr(i,:);
    end
    if Ytr(i)==4
        M4=M4+1;
        S_4(M4,:)=Xtr(i,:);
    end
    if Ytr(i)==5
        M5=M5+1;
        S_5(M5,:)=Xtr(i,:);
    end
    if Ytr(i)==6
        M6=M6+1;
        S_6(M6,:)=Xtr(i,:);
    end
    if Ytr(i)==7
        M7=M7+1;
        S_7(M7,:)=Xtr(i,:);
    end
    if Ytr(i)==8
        M8=M8+1;
        S_8(M8,:)=Xtr(i,:);
    end
    if Ytr(i)==9
        M9=M9+1;
        S_9(M1,:)=Xtr(i,:);
    end
end

for i=1:10000
    if Yte(i)==0
        M0=M0+1;
        S_0(M0,:)=Xte(i,:);
    end
    if Yte(i)==1
        M1=M1+1;
        S_1(M1,:)=Xte(i,:);
    end
    if Yte(i)==2
        M2=M2+1;
        S_2(M2,:)=Xte(i,:);
    end
    if Yte(i)==3
        M3=M3+1;
        S_3(M3,:)=Xte(i,:);
    end
    if Yte(i)==4
        M4=M4+1;
        S_4(M4,:)=Xte(i,:);
    end
    if Yte(i)==5
        M5=M5+1;
        S_5(M5,:)=Xte(i,:);
    end
    if Yte(i)==6
        M6=M6+1;
        S_6(M6,:)=Xte(i,:);
    end
    if Yte(i)==7
        M7=M7+1;
        S_7(M7,:)=Xte(i,:);
    end
    if Yte(i)==8
        M8=M8+1;
        S_8(M8,:)=Xte(i,:);
    end
    if Yte(i)==9
        M9=M9+1;
        S_9(M9,:)=Xte(i,:);
    end
end
%% SECTION 2 CONSTRUCTING files Tr (training images, 5000 for each digit) and Te (test images,
% 1000 for each digit with numbers 5001:6000 for each digit)




Tr=zeros(50000,784);
Tr(1:5000,:)=S_0(1:5000,:);
Tr(5001:10000,:)=S_1(1:5000,:);
Tr(10001:15000,:)=S_2(1:5000,:);
Tr(15001:20000,:)=S_3(1:5000,:);
Tr(20001:25000,:)=S_4(1:5000,:);
Tr(25001:30000,:)=S_5(1:5000,:);
Tr(30001:35000,:)=S_6(1:5000,:);
Tr(35001:40000,:)=S_7(1:5000,:);
Tr(40001:45000,:)=S_8(1:5000,:);
Tr(45001:50000,:)=S_9(1:5000,:);

Te=zeros(10000,784);
Te(1:1000,:)=S_0(5001:6000,:);
Te(1001:2000,:)=S_1(5001:6000,:);
Te(2001:3000,:)=S_2(5001:6000,:);
Te(3001:4000,:)=S_3(5001:6000,:);
Te(4001:5000,:)=S_4(5001:6000,:);
Te(5001:6000,:)=S_5(5001:6000,:);
Te(6001:7000,:)=S_6(5001:6000,:);
Te(7001:8000,:)=S_7(5001:6000,:);
Te(8001:9000,:)=S_8(5001:6000,:);
Te(9001:10000,:)=S_9(5001:6000,:);

%% SECTION 3 CALCULATING DISTANCES to each of 10 classes for all 784 windows  of size K
K=11; % window size
K2=round((K-1)/2);
W11_65=zeros(10000,10,28,28);
T=0;
tic
for s1=1:28
    for s2=1:28
        T=T+1

        % constructing restrictions on the windows of the training images
        WTr=zeros(50000,K*K);
        for i=1:50000
            X=reshape(Tr(i,:),28,28)';
            XX=zeros(28+K-1,28+K-1);
            XX(K2+1:K2+28,K2+1:K2+28)=X;
            U=XX(s1:s1+K-1,s2:s2+K-1);
            WTr(i,:)=reshape(U',1,K*K);
        end
        
        % constructing the restriction of test images for the window
        WTe=zeros(10000,K*K);
        for i=1:10000
            Y=reshape(Te(i,:),28,28)';
            YY=zeros(28+K-1,28+K-1);
            YY(K2+1:K2+28,K2+1:K2+28)=Y;
            W=YY(s1:s1+K-1,s2:s2+K-1);
            WTe(i,:)=reshape(W',1,K*K);
        end
        
        % computing the distances in the power 2 between test images and training classes on the windows
        D=pdist2(WTe,WTr,'squaredeuclidean');
        for i=10000:10000
            for j=1:10
                W11_65(i,j,s1,s2)=min(D(i,(5000*(j-1))+1:5000*j));
            end
        end
    end
end

toc
%% SECTION 3 NUMBER of ERRORS
save('W11_65.mat','W11_65')

load('JJ_65.mat','JJ_65')
U=zeros(10000,10);
for i=1:10000
    for j=1:10
U(i,j)=sum(sum(W11_65(i,j,:,:)));
    end
end
M=0;
for i=1:10000
    [a,b]=min(U(i,:));
    if JJ_65(i)==b
    else
        M=M+1;
    end
end
NUMBER_of_ERRORS_TEST_SET=M


