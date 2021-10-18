clear all
close all
clc

n = 130;

load pairs_golden_standard;
pairs = pairs_golden_standard;
load self_golden_standard;
self = self_golden_standard;


T=1210;
expression = zeros(n, T);


rng(200) 
expression(1:n,1)=normrnd(0,1,n,1);
expression(1:n,2)=normrnd(0,1,n,1);
expression(1:n,3)=normrnd(0,1,n,1);
expression(1:n,4)=normrnd(0,1,n,1);
expression(1:n,5)=normrnd(0,1,n,1);



a = exp((1/6)*pi*1.0i);
b = exp(-(1/6)*pi*1.0i);
c = a+b;
d = a*b;

for i=6:T
    for j=1:n
        expression(j,i) = normrnd(0, 1);


        array = find(pairs(:,2)==j);


        if size(array, 1)==0
            index = find(self(:,2)==j);
            factor = self(index, 4);
            vec = expression(j,:);            
         
            expression(j,i) = expression(j,i) +  atan((c*self(index, 4)*expression(j,i-1) + d * self(index,4) * self(index,4) * expression(j,i-5))^2);            
          

        else

            for k=1:size(array,1)
                previous_value = pairs(array(k),4)*expression(pairs(array(k),1), i-pairs(array(k), 3));              
                 expression(j,i) =  expression(j,i) + log(previous_value^2^2) + tanh(previous_value*expression(j,i-pairs(array(k),3)));
            end
            
             if size(array,1) > 1
                 previous_value1  = expression(pairs(array(1),1), i-pairs(array(1), 3));
                previous_value2  = expression(pairs(array(2),1), i-pairs(array(2), 3));
                expression(j,i) = expression(j,i) + sin((previous_value1*previous_value2)^2);
            end
            
        end
    end
end

%no noise
expression0 = expression(:,11:end);


[expression0] = standalization(expression0);


figure(1)
plot(expression0)


expression = expression0;
save filter_norm_expression0.mat expression;

