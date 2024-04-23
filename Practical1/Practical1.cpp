#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

// Structure to represent a graph
struct Graph {
    int V; // Number of vertices
    vector<vector<int>> adj; // Adjacency list

    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    // Add an edge between two vertices
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); // Assuming undirected graph
    }

    // Parallel Breadth-First Search
    void parallelBFS(int startVertex) {
        vector<bool> visited(V, false);
        queue<int> q;
        q.push(startVertex);
        visited[startVertex] = true;

        while (!q.empty()) {
            int currentVertex = q.front();
            q.pop();
            cout << currentVertex << " ";

            #pragma omp parallel for
            for (int i = 0; i < adj[currentVertex].size(); ++i) {
                int adjacentVertex = adj[currentVertex][i];
                if (!visited[adjacentVertex]) {
                    visited[adjacentVertex] = true;
                    q.push(adjacentVertex);
                }
            }
        }
        cout << endl;
    }

    // Parallel Depth-First Search
    void parallelDFSUtil(int v, vector<bool>& visited) {
        cout << v << " ";
        visited[v] = true;

        #pragma omp parallel for
        for (int i = 0; i < adj[v].size(); ++i) {
            int adjacentVertex = adj[v][i];
            if (!visited[adjacentVertex])
                parallelDFSUtil(adjacentVertex, visited);
        }
    }

    void parallelDFS(int startVertex) {
        vector<bool> visited(V, false);
        parallelDFSUtil(startVertex, visited);
        cout << endl;
    }
};

int main() {
    // Create a graph
    Graph g(6);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);

    cout << "Parallel Breadth-First Search: ";
    g.parallelBFS(0);

    cout << "Parallel Depth-First Search: ";
    g.parallelDFS(0);

    return 0;
}
