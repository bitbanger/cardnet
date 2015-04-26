import math, random, string, sys
from os import listdir
from os.path import isfile, join

# Returns a random real value in the range [-a, a]
def rand_neg(a):
	return (2*a)*random.random() - a

def sig(n):
	n = min(500, max(-500, n))
	try:
		return 1.0/(1+math.exp(-n))
	except:
		print n
		return 1.0/(1+math.exp(-n))

def dsig(n):
	sv = sig(n)
	return sv*(1-sv)

# Knobs
precociousness = 0.15
epochs = 100

real_train = False
real_test = False

# suit and val for four cards, plus a bias
num_in = 20 + 1
num_hid = 30
# one for suit and 13 to pick from for val
num_out = 17

# Weights
in_hid_weights = [[rand_neg(1) for j in range(num_hid)] for i in range(num_in)]
hid_out_weights = [[rand_neg(1) for j in range(num_out)] for i in range(num_hid)]

# Values (temporary)
in_vals = [1.0] * num_in
hid_vals = [1.0] * num_hid
out_vals = [1.0] * num_out

def serialize(fname):
	with open(fname, 'w+') as f:
		f.write("%d %d %d\n" % (num_in, num_hid, num_out))
		for m in in_hid_weights:
			for w in m:
				f.write("%f " % (w))
		f.write("\n")
		for m in hid_out_weights:
			for w in m:
				f.write("%f " % (w))

# Prop function for single example
def learn(ins):
	if len(ins) != num_in-1:
		print "need %d inputs, got %d; will not learn this!" % (num_in-1, len(ins))
	
	# Copy example values into input neurons
	for i in range(len(ins)):
		in_vals[i] = ins[i]
	
	# Propagate them forward to the hidden neurons
	for i in range(len(hid_vals)):
		# Weighted sum of all input values, including bias
		in_sum = sum([(in_vals[j]*in_hid_weights[j][i]) for j in range(len(in_vals))])
		
		# The hidden neuron value becomes that sigmoid function's value
		hid_vals[i] = sig(in_sum)
	
	# Propagate hidden values forward to output neurons
	for i in range(len(out_vals)):
		# Weighted sum of all hidden values
		hid_sum = sum([(hid_vals[j]*hid_out_weights[j][i]) for j in range(len(hid_vals))])
		
		# The output neuron value becomes that sigmoid function's value
		out_vals[i] = sig(hid_sum)

# Backprop function to correct weights
def backprop(outs, precociousness):
	if len(outs) != num_out:
		print "need %d target outputs, got %d; will not backprop this!" % (num_out, len(outs))
	
	# Vector of output neuron errors (simple list comp)
	d_out_hat = [dsig(out_vals[i]) * (outs[i] - out_vals[i]) for i in range(num_out)]
	
	# Vector of hidden neuron errors (weighted dependencies to output; inelegant! :( )
	d_hid_hat = [0.0] * num_hid
	for i in range(num_hid):
		# The error is the weighted sum of the output errors wired to this hidden neuron
		error = sum([d_out_hat[j]*hid_out_weights[i][j] for j in range(num_out)])
		
		# Multiply that by the derivative of the sigmoid function evaluated at the hidden value
		d_hid_hat[i] = dsig(hid_vals[i]) * error
	
	# Re-weight wires from hidden to output
	for i in range(num_hid):
		for j in range(num_out):
			# Learning rate times hidden neuron's value times output neuron error
			hid_out_weights[i][j] += precociousness * hid_vals[i] * d_out_hat[j]
	
	# Re-weight wires from input to hidden
	for i in range(num_in):
		for j in range(num_hid):
			in_hid_weights[i][j] += precociousness * in_vals[i] * d_hid_hat[j]
	
	# Return squared error
	return sum([0.5 * (outs[i] - out_vals[i])**2 for i in range(num_out)])

table = {
	1: [0, 1, 2],
	2: [0, 2, 1],
	3: [1, 0, 2],
	4: [1, 2, 0],
	5: [2, 0, 1],
	6: [2, 1, 0],
}
def sort_trick(hand, real = False):
	c1 = None
	c2 = None
	shand = sorted(hand)
	for i in range(len(shand)-1):
		if shand[i][0] == shand[i+1][0]:
			c1, c2 = shand[i], shand[i+1]
			break

	mn, mx = min(c1, c2, key = lambda x: x[1]), max(c1, c2, key = lambda x: x[1])
	dist = mx[1] - mn[1]

	handed = mn
	guessed = mx
	if dist > 6:
		handed = mx
		guessed = mn
		dist = 13 - dist

	to_perm = sorted([x for x in hand if (x != guessed and x != handed)], key = lambda x: x[::-1])
	perm = table[dist]
	in_order = [to_perm[i] for i in perm]

	if real:
		return [handed] + in_order + [guessed]
	else:
		return [handed] + perm + [guessed]

def flatten(l):
	r = []
	for t in l:
		rp1 = [0.0] * 4
		rp1[t[0]-1] = 1.0

		rp2 = [0.0] * 13
		rp2[t[1]-1] = 1.0
		
		r += rp1
		r += rp2
	return r

suit_names = {
	1: "clubs",
	2: "diamonds",
	3: "hearts",
	4: "spades",
}
card_names = {

}
	

def main():
	cards = [(suit, val) for suit in range(1, 5) for val in range(1, 14)]
	
	for i in range(10):
		hand = random.sample(cards, 5)
		print "got: %s" % hand
		trick = sort_trick(hand)
		print "made: %s" % trick

	training = []
	for i in range(100000):
		training.append(sort_trick(random.sample(cards, 5), real = real_train))
	test = []
	for i in range(1000):
		test.append(sort_trick(random.sample(cards, 5), real = real_test))

	last_percent = -1.0
	correctness = 0.0
	
	for e in range(epochs):
		random.shuffle(training)
		for tr in training:
			# to_learn = flatten(tr[:-1])
			val = [0.0] * 13
			val[tr[0][1]-1] = 1.0
			suit = [0.0] * 4
			suit[tr[0][0]-1] = 1.0
			to_learn = None
			if real_train:
				to_learn = suit + val + [x[1]*1.0/13 for x in tr[1:-1]]
			else:
				to_learn = suit + val + [x*1.0/2 for x in tr[1:-1]]
			learn(to_learn)

			want_suit = [0.0] * 4
			want_suit[tr[0][0]-1] = 1.0
			#want_suit = tr[0][0]*1.0 / 4

			want_val = [0.0] * 13
			want_val[tr[-1][1]-1] = 1.0
			#want_val = tr[-1][1]*1.0 / 13

			#want = [want_suit] + want_val
			want = want_suit + want_val

			backprop(want, precociousness)

			# backprop([tr[-1][0]*1.0/4, tr[-1][1]*1.0/13], precociousness)
			percent = int(e*100.0/epochs)
			if percent != last_percent:
				last_percent = percent
				sys.stdout.write("Status: %d%% done training (%.4f)            \r" % (percent, correctness))
				sys.stdout.flush()
		correctness = ctest(test)

def ctest(test):
	correct = 0
	for t in test:
		val = [0.0] * 13
		val[t[0][1]-1] = 1.0
		suit = [0.0] * 4
		suit[t[0][0]-1] = 1.0
		to_learn = None
		if real_test:
			to_learn = suit + val + [x[1]*1.0/13 for x in t[1:-1]]
		else:
			to_learn = suit + val + [x*1.0/2 for x in t[1:-1]]
		learn(to_learn)
		sv = out_vals[:4]
		ov = out_vals[4:]
		suit_guess = sv.index(max(sv)) + 1
		guess = ov.index(max(ov)) + 1
		if guess == t[-1][1] and suit_guess == t[-1][0]:
		# if guess == t[-1][1]:
		# if suit_guess == t[-1][0]:
			correct += 1
		
	return correct*1.0/len(test)

if __name__ == '__main__':
	main()
