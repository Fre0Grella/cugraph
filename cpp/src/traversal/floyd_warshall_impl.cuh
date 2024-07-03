/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "prims/count_if_e.cuh"
#include "prims/fill_edge_src_dst_property.cuh"
#include "prims/reduce_op.cuh"
#include "prims/transform_reduce_e.cuh"
#include "prims/transform_reduce_v_frontier_outgoing_e_by_dst.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "prims/update_v_frontier.cuh"
#include "prims/vertex_frontier.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/vertex_partition_device_view.cuh>
#include <cugraph/cpp/src/c_api/strongly_connected_components.cpp>
#include <cugraph_c/labeling_algorithms.h>
#include <graph.hpp>
#include <graph_view.hpp>


#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/optional.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <limits>

namespace cugraph {
namespace detail {
    // Function to check if a graph is strongly connected
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed>
bool is_strongly_connected(
        const raft::handle_t& handle,
        graph_view_t<vertex_t, edge_t, false, multi_gpu>& graph_view)
{
    cugraph_labeling_result_t* result = nullptr;
    cugraph_error_t* error = nullptr;

    // create graph_t from the view
    auto graph = cugraph::c_api::graph_from_view(handle, graph_view);

    // Check if graph is scc with the functor declare in
    cugraph::c_api::scc_functor functor(&handle, graph, false);
    functor.template operator()<vertex_t, edge_t, weight_t, false, true>();
    result = functor.result_;

    if (result == nullptr) {
        return false;
    }

    auto num_vertices = graph_view.get_number_of_vertices();
    rmm::device_uvector<vertex_t> components(num_vertices, handle.get_stream());
    raft::update_device(components.data(), result->components_->data(), num_vertices, handle.get_stream());

    vertex_t first_component;
    raft::update_host(&first_component, components.data(), 1, handle.get_stream());
    handle.sync_stream();

    // Check if all the components are equal
    bool all_same = thrust::all_of(
        rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
        components.begin(),
        components.end(),
        all_equal{first_component}
    );

    // Clean memory
    cugraph::c_api::cugraph_free_labeling_result(result);
    return all_same;
}

} // end namespace detail

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed>
void floyd_warshall(
        const raft::handle_t& handle,
        graph_view_t<vertex_t, edge_t, weight_t, store_transposed>& graph_view)
{

};

} // end namesapce cugraph