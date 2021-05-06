clear all
close all
%% SECTION 1 CONSTRUCTION of VALIDATION and TEST SETS
tic
load('W11_65.mat','W11_65')

n=784;
d=load('mnist.mat');
Xtr=double(d.trainX);
Ytr=double(d.trainY);
Xte=double(d.testX);
Yte=double(d.testY);
Xtr1=zeros(28,28,1,60000);
Xte1=zeros(28,28,1,10000);
S1=zeros(1,784);
S2=zeros(1,784);
S3=zeros(1,784);
S4=zeros(1,784);
S5=zeros(1,784);
S6=zeros(1,784);
S7=zeros(1,784);
S8=zeros(1,784);
S9=zeros(1,784);
S10=zeros(1,784);
M1=0;
M2=0;
M3=0;
M4=0;
M5=0;
M6=0;
M7=0;
M8=0;
M9=0;
M10=0;
for i=1:60000
    if Ytr(i)==0
        M1=M1+1;
        S1(M1,:)=Xtr(i,:);
    end
    if Ytr(i)==1
        M2=M2+1;
        S2(M2,:)=Xtr(i,:);
    end
    if Ytr(i)==2
        M3=M3+1;
        S3(M3,:)=Xtr(i,:);
    end
    if Ytr(i)==3
        M4=M4+1;
        S4(M4,:)=Xtr(i,:);
    end
    if Ytr(i)==4
        M5=M5+1;
        S5(M5,:)=Xtr(i,:);
    end
    if Ytr(i)==5
        M6=M6+1;
        S6(M6,:)=Xtr(i,:);
    end
    if Ytr(i)==6
        M7=M7+1;
        S7(M7,:)=Xtr(i,:);
    end
    if Ytr(i)==7
        M8=M8+1;
        S8(M8,:)=Xtr(i,:);
    end
    if Ytr(i)==8
        M9=M9+1;
        S9(M9,:)=Xtr(i,:);
    end
    if Ytr(i)==9
        M10=M10+1;
        S10(M10,:)=Xtr(i,:);
    end
end
%%%%%%%%%%%
for i=1:10000
    if Yte(i)==0
        M1=M1+1;
        S1(M1,:)=Xte(i,:);
    end
    if Yte(i)==1
        M2=M2+1;
        S2(M2,:)=Xte(i,:);
    end
    if Yte(i)==2
        M3=M3+1;
        S3(M3,:)=Xte(i,:);
    end
    if Yte(i)==3
        M4=M4+1;
        S4(M4,:)=Xte(i,:);
    end
    if Yte(i)==4
        M5=M5+1;
        S5(M5,:)=Xte(i,:);
    end
    if Yte(i)==5
        M6=M6+1;
        S6(M6,:)=Xte(i,:);
    end
    if Yte(i)==6
        M7=M7+1;
        S7(M7,:)=Xte(i,:);
    end
    if Yte(i)==7
        M8=M8+1;
        S8(M8,:)=Xte(i,:);
    end
    if Yte(i)==8
        M9=M9+1;
        S9(M9,:)=Xte(i,:);
    end
    if Yte(i)==9
        M10=M10+1;
        S10(M10,:)=Xte(i,:);
    end
end
Sval=zeros(10000,784);
Sval(1:1000,:)=S1(5001:6000,:);
Sval(1001:2000,:)=S2(5001:6000,:);
Sval(2001:3000,:)=S3(5001:6000,:);
Sval(3001:4000,:)=S4(5001:6000,:);
Sval(4001:5000,:)=S5(5001:6000,:);
Sval(5001:6000,:)=S6(5001:6000,:);
Sval(6001:7000,:)=S7(5001:6000,:);
Sval(7001:8000,:)=S8(5001:6000,:);
Sval(8001:9000,:)=S9(5001:6000,:);
Sval(9001:10000,:)=S10(5001:6000,:);

Stest=zeros(1,784);
Stest(1:M1-6000,:)=S1(6001:M1,:);
a=size(Stest,1);
Stest(a+1:a+M2-6000,:)=S2(6001:M2,:);
a=size(Stest,1);
Stest(a+1:a+M3-6000,:)=S3(6001:M3,:);
a=size(Stest,1);
Stest(a+1:a+M4-6000,:)=S4(6001:M4,:);
a=size(Stest,1);
Stest(a+1:a+M5-6000,:)=S5(6001:M5,:);
a=size(Stest,1);
Stest(a+1:a+M6-6000,:)=S6(6001:M6,:);
a=size(Stest,1);
Stest(a+1:a+M7-6000,:)=S7(6001:M7,:);
a=size(Stest,1);
Stest(a+1:a+M8-6000,:)=S8(6001:M8,:);
a=size(Stest,1);
Stest(a+1:a+M9-6000,:)=S9(6001:M9,:);
a=size(Stest,1);
Stest(a+1:a+M10-6000,:)=S10(6001:M10,:);




TIME=toc




%% Section 1
SvalNEW=zeros(10000,784);
for i=1:1000
    for j=1:10 
        SvalNEW(10*(i-1)+j,:)=Sval((j-1)*1000+i,:);
    end
end

JJ65_new=zeros(1,10000);
for i=1:10000
    a=ceil(i/10);
    b=i-(a-1)*10;
    JJ65_new(i)=b;
end

%% SECTION 2 APPROXIMATION of DISTANCE MATRIX W11_75 by NEURAL NETWORKS
A11_75=zeros(10000,10,28,28);

for s1=1:28
    
for s2=1:28
    s1
    s2
    TIME=toc
    
    
    
        
      Xtest=zeros(1,121,1,10000);
      for i=1:10000
      X=reshape(Stest(i,:),28,28)';
      XX=zeros(38,38);
      XX(6:33,6:33)=X;
      U=XX(s1:s1+10,s2:s2+10);
      Xtest(1,:,1,i)=reshape(U',1,121)/255;
      end      
 for k=1:10
  
  Xtrain=zeros(1,121,1,10000);
  Ytrain=zeros(1,10000);
  
  
  for i=1:10000
      
      X=reshape(SvalNEW(i,:),28,28)';
      XX=zeros(38,38);
      XX(6:33,6:33)=X;
      U=XX(s1:s1+10,s2:s2+10);
      Xtrain(1,:,1,i)=reshape(U',1,121)/255;
      a=ceil(i/10);
      b=i-(a-1)*10;
      Ytrain(i)=W11_65((b-1)*1000+a,k,s1,s2);
  end
  
           Ytrain=0.000001*Ytrain;
          
    

         
 N=40;


layer1=imageInputLayer([1 121 1],'Normalization','none');
layer2=fullyConnectedLayer(N,'WeightLearnRateFactor',1,'BiasLearnRateFactor',1);
layer3=reluLayer();
layer4=batchNormalizationLayer();
layer5=fullyConnectedLayer(1,'WeightLearnRateFactor',1,'BiasLearnRateFactor',1);
layer6=regressionLayer();

layers=[layer1,layer2,layer3,layer4,layer5,layer6];


 options=trainingOptions('sgdm','MaxEpochs',500,'MiniBatchSize',1000,'Verbose',0,...
    'InitialLearnRate',0.1);

net=trainNetwork(Xtrain,(Ytrain)',layers,options);

Y=predict(net, Xtest);
A11_75(:,k,s1,s2)=Y;

 end
end
save('A11_75.mat','A11_75')
end

%% SECTION 3 NUMBER OF ERRORS
load('JJ_75.mat','JJ_75')
U=zeros(10000,10);
for i=1:10000
    for j=1:10
U(i,j)=sum(sum(A11_75(i,j,:,:)));
    end
end
M=0;
for i=1:10000
    [a,b]=min(U(i,:));
    if JJ_75(i)==b
    else
        M=M+1;
    end
end
NUMBER_of_ERRORS_TEST_SET=M




