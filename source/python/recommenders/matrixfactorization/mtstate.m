function state = mtstate(seed);

state = uint32(zeros(625,1));

state(1) = uint32(seed);
for i=1:623,
   tmp = uint64(1812433253)*uint64(bitxor(state(i),bitshift(state(i),-30)))+i;
   state(i+1) = uint32(bitand(tmp,uint64(intmax('uint32'))));
end
state(625) = 1;