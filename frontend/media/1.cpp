//#include<iostream>
//#include<vector>
//#include<algorithm>
//
//using namespace std;
//
//struct Edge {
//	int id;
//	int u, v;
//	int weight;
//	Edge(int id, int u, int v, int weight) :id(id), u(u), v(v), weight(weight) {
//	}
//};
//class Unionfind {
//public:
//	Unionfind(int n) :parent(n), rank(n, 0) {
//		for (int i = 0; i < n; i++) {
//			parent[i] = i;
//		}
//	}
//	int find(int x) {
//		if (x != parent[x])
//			parent[x] = find(parent[x]);
//		return parent[x];
//	}
//	void unite(int x, int y) {
//		int rootX = find(x);
//		int rootY = find(y);
//		if (rootX != rootY) {
//			if (rank[rootX] < rank[rootY]) {
//				parent[rootX] = rootY;
//			}
//			else if (rank[rootX] > rank[rootY]) {
//				parent[rootY] = rootX;
//			}
//			else {
//				parent[rootY] = rootX;
//				rank[rootX]++;
//			}
//		}
//	}
//private:
//	vector<int>parent;
//	vector<int>rank;
//};
//bool compare(const Edge& e1, const Edge& e2) {
//	return e1.weight < e2.weight;
//}
//int main() {
//	int n, e;
//	cin >> n >> e;
//	vector<Edge>edges;
//	for (int i = 0; i < e; ++i) {
//		int id, u, v, weight;
//		cin >> id >> u >> v >> weight;
//		edges.push_back(Edge(id, u, v, weight));
//	}
//	sort(edges.begin(), edges.end(), compare);
//	Unionfind uf(n);
//	vector<int>mstEdges;
//	int totalweight = 0;
//	for (const auto& edge : edges) {
//		if (uf.find(edge.u) != uf.find(edge.v)) {
//			uf.unite(edge.u, edge.v);
//			mstEdges.push_back(edge.id);
//			totalweight += edge.weight;
//		}
//	}
//	sort(mstEdges.begin(), mstEdges.end());
//	for (int id : mstEdges) {
//		cout << id << " ";
//	}
//	cout << endl;
//	cout << totalweight;
//	return 0;
//}