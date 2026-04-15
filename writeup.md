## ZelusBench

> Can the model focus on what matters and ignore what doesn't?

But how does one quantify what *matters*? Our goal was to build a grounded attention benchmark,
one where *whether information matters or doesnt or is shifted* is deterministically quantifable.

In order to acieve this we build ZelusBench -- an attention benchmark, that uses spatial reasoning. 
The premise is simple, build a represetation of 2/3D space using a range of standard geometry technicuqes, 
the probe questions about the space.

The benefit of working in a space, is it is easy interpretable, allowing us to easliy define the otherwise
tricky concept that is "what information matters". 

The benchmark generates a space containing points, each defined in relation to others -- so called "chains". For example A is x units
from the origin and so forth. At the end (or even during), the model recieves "queries" where it must answer some 
question about the space, eg

- is A closer to B or C
- what are the coordiantes of D

Each of which having a true, deterministic value. By tweaking parts of the generation, we can probe the LLMs attention
abilities. 

- Selective -- add noise (more points relative depth of chains)
- Sustained -- longer chain depths
- Shifting -- redefining points (translation/relflection/rotation etc)

By varying the paramters as described above, we can probe the llms ability in each scenario...