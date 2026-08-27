import numpy as np;

class Node: 
    def __init__(self):
        # the vector stored at this node
        self.data = np.random.randn(20);

        # weights governing this node interacts with other nodes
        self.wkey = np.random.randn(20, 20);
        self.wquery = np.random.randn(20, 20);
        self.wvalue = np.random.randn(20, 20);

    def key(self):
        # what do I have?
        return self.wkey @ self.data;

    def query(self):
        # what am I looking for?
        return self.wquery @ self.data;

    def value(self):
        # what do I publically reveal/broadcast to others?
        return self.wvalue @ self.data;

class Graph:
    def __init__(self):
        # make 10 nodes
        self.nodes = [Node() for _ in range(10)];
        # make 40 edges
        random_i = lambda: np.random.randint(len(self.nodes));
        self.edges = [[random_i()  , random_i()] for _ in range(40)];


    def run(self):
        updates = [];
        for i , n in enumerate(self.nodes):
            # what is this node looking for?
            q = n.query();

            inputs = [self.nodes[ifrom] for [ifrom , ito ] in self.edges if ito ==i]
            if len(inputs) == 0:
                continue;
            # gather their keys i.e what they hold
            keys = [m.key() for m in inputs];

            # calculate the compatibilities
            scores = [k.dot(q) for k in keys];

            # softmax them so they sum to 1
            scores = np.exp(scores);
            # gather the appropriate values with a weighted sum
            values = [m.value() for m in inputs];
            update = sum([s*v for s, v in zip(scores, values)]);
            updates.append(update);

            # testing information
            print(f"\nNode {i}")
            print(f"  Incoming nodes: {len(inputs)}")
            print(f"  Attention scores: {scores}")
            print(f"  Sum of scores: {np.sum(scores):.4f}")
            print(f"  Update shape: {update.shape}")

        for n, u in zip(self.nodes , updates):
            n.data = n.data + u #residual connection


print("Creating graph...")

g = Graph()

print("\nGraph created.")
print("Number of nodes:", len(g.nodes))
print("Number of edges:", len(g.edges))


# Check initial node data
print("\nInitial Node 0 data:")
print(g.nodes[0].data)

print("\nInitial Node 0 data shape:")
print(g.nodes[0].data.shape)


# Save a copy before attention
old_data = [
    node.data.copy()
    for node in g.nodes
]


# Run attention
print("\n==============================")
print("Running attention...")
print("==============================")

g.run()


# Check after attention
print("\n==============================")
print("After attention")
print("==============================")

print("\nNode 0 data:")
print(g.nodes[0].data)

print("\nNode 0 data shape:")
print(g.nodes[0].data.shape)


# Check whether node data changed
print("\n==============================")
print("Testing updates")
print("==============================")

for i, node in enumerate(g.nodes):

    changed = not np.array_equal(
        old_data[i],
        node.data
    )

    print(f"Node {i}: data changed = {changed}")