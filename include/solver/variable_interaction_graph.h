#ifndef CONTINUOUS_OPTIMIZATION_VARIABLE_INTERACTION_GRAPH_H
#define CONTINUOUS_OPTIMIZATION_VARIABLE_INTERACTION_GRAPH_H

#include <iostream>
#include <vector>
#include <deque>
#include <unordered_set>

using namespace std;

namespace solver {

    enum class graph_type { Fully_Connected, Fully_Disconnected, Is_Connected };

    class variable_interaction_graph{
    protected:
        vector<unordered_set<size_t>> graph;
        vector<bool> vertices_active;
        size_t num_vertices;
        size_t num_edges;
        graph_type graph_type_vig;

    public:
        variable_interaction_graph() {
            num_vertices = 0;
            num_edges = 0;
        }

        explicit variable_interaction_graph(size_t v){
            graph.reserve(v);
            vertices_active.reserve(v);
            num_edges = 0;
            num_vertices = v;
            for(size_t i = 0; i < v; i++){
                graph.emplace_back();
                vertices_active.push_back(true);
            }
        }

        void add_edge(size_t source, size_t dest){
            if(source < 0 || source > num_vertices-1)
                throw std::logic_error("Invalid source vertex.");
            if(dest < 0 || dest > num_vertices-1)
                throw std::logic_error("Invalid destination vertex.");
            if(graph[source].find(dest) == graph[source].end() && source != dest){
                graph[source].insert(dest);
                graph[dest].insert(source);
                ++num_edges;
            }
        }

        size_t get_number_adjacent_vertices(size_t vertex){
            return (vertices_active[vertex] ? graph[vertex].size() : 0);
        }

        unordered_set<size_t> get_adjacent_vertices(size_t vertex){
            if(vertex < 0 || vertex > num_vertices-1)
                throw std::logic_error("Invalid source vertex.");
            unordered_set<size_t> adj;
            adj.reserve(graph[vertex].size());
            for(size_t i : graph[vertex]){
                if(vertices_active[i])
                    adj.insert(i);
            }
            return adj;
        }

        void get_adjacent_vertices(unordered_set<size_t> &vertices_sub_graph, deque<size_t> &vertices_to_visit, size_t vertex){
            if(vertex < 0 || vertex > num_vertices-1)
                throw std::logic_error("Invalid source vertex.");
            for(size_t i : graph[vertex]){
                if(vertices_active[i] && vertices_sub_graph.find(i) == vertices_sub_graph.end()) {
                    vertices_sub_graph.insert(i);
                    vertices_to_visit.push_back(i);
                }
            }
        }

        template<typename vector_t>
        vector<unordered_set<size_t>> generate_recombination_graph(const vector_t p1, const vector_t p2){
            vector<unordered_set<size_t>> sub_graphs;
            sub_graphs.reserve(num_vertices);
            size_t inactive_vertices = 0;
            size_t cont_vertices = 0;
            for(size_t i = 0; i < num_vertices; i++){
                if(p1[i] == p2[i]){
                    vertices_active[i] = false;
                    inactive_vertices++;
                }
            }
            if(inactive_vertices < num_vertices){
                for(size_t i = 0; i < num_vertices; i++){
                    unordered_set<size_t> vertices_sub_graph;
                    vertices_sub_graph.reserve(num_vertices);
                    if(vertices_active[i]){
                        deque<size_t> vertices_to_visit;
                        vertices_to_visit.push_back(i);
                        vertices_sub_graph.insert(i);
                        while(!vertices_to_visit.empty()){
                            vertices_active[vertices_to_visit[0]] = false;
                            get_adjacent_vertices(vertices_sub_graph, vertices_to_visit, vertices_to_visit[0]);
                            if(vertices_sub_graph.size() + cont_vertices + inactive_vertices == num_vertices){
                                i = num_vertices;
                                vertices_to_visit.clear();
                            }
                            else {
                                vertices_to_visit.pop_front();
                            }
                        }
                    }
                    if(!vertices_sub_graph.empty()){
                        sub_graphs.push_back(vertices_sub_graph);
                        cont_vertices += vertices_sub_graph.size();
                    }
                }
            }
            for(size_t i = 0; i < num_vertices; i++){
                vertices_active[i] = true;
            }
            return sub_graphs;
        }

        void pre_processing_graph(){
            if(num_edges == 0){
                graph_type_vig = graph_type::Fully_Disconnected;
            }
            else if(num_edges == (num_vertices * (num_vertices - 1) / 2)){
                graph_type_vig = graph_type::Fully_Connected;
            }
            else{
                graph_type_vig = graph_type::Is_Connected;
            }
        }

        void print_vertices(){
            cout << "number of vertices: " << graph.size() << endl;
            cout << "vertices >> {";
            for(size_t i = 0; i < num_vertices-1; i++){
                cout << i << "; ";
            }
            cout << num_vertices-1 << "}" << endl;
        }

        void print_edges() {
            cout << "edges >> {";
            for(size_t index_source = 0; index_source < num_vertices; index_source++){
                if(!graph[index_source].empty()) {
                    for(auto i = graph[index_source].begin(); i != graph[index_source].end(); i++){
                        cout << " {" << index_source << "," << *i << "}";
                    }
                }
            }
            cout << " }" << endl;
        }

        size_t size_vertices() const{
            return num_vertices;
        }

        size_t size_edges() const {
            return num_edges;
        }

        graph_type get_graph_type(){
            return graph_type_vig;
        }
    };

}

#endif