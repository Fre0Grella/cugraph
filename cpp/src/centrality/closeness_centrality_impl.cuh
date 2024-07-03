/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

//include prims

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/core/handle.hpp>

#include <thrust/functional.h>
#include <thrust/optional.h>
#include <thrust/reduce.h>

namespace cugraph {
namespace detail {
  template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          typename VertexIterator>
rmm::device_uvector<weight_t> closeness_centrality(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  VertexIterator vertices_begin,
  VertexIterator vertices_end,
  bool const normalized,
  bool const include_self_loop,
  bool const do_expensive_check)
  {
    //
    //  Closeness Centrality algorithm based on Katz and Kider version of paralell Floyd-Warshall (2008)  
    //
    if (do_expensive_check) { //Check graph integrity
      auto vertex_partition =
        vertex_partition_device_view_t<vertex_t, multi_gpu>(graph_view.local_vertex_partition_view());
      auto num_invalid_vertices =
        thrust::count_if(handle.get_thrust_policy(),
                         vertices_begin,
                         vertices_end,
                         [vertex_partition] __device__(auto val) {
                           return !(vertex_partition.is_valid_vertex(val) &&
                                    vertex_partition.in_local_vertex_partition_range_nocheck(val));
                         });
      if constexpr (multi_gpu) {
        num_invalid_vertices = host_scalar_allreduce(
          handle.get_comms(), num_invalid_vertices, raft::comms::op_t::SUM, handle.get_stream());
      }
      CUGRAPH_EXPECTS(num_invalid_vertices == 0,
                    "Invalid input argument: sources have invalid vertex IDs.");
    }
  
  //Allocate centralities weights and number of vertices
  rmm::device_uvector<weight_t> centralities(graph_view.number_of_vertices(),
                                             handle.get_stream());
  size_t vertex_num = thrust::distance(vertices_begin, vertices_end);

  //TODO: to implement
  centralities = floyd_warshall();
  
  if (normalized) {
    weight_t n = static_cast<weight_t>(graph_view.number_of_vertices());
    if (!include_self_loop)
      n = n - 1;
    
    rmm::device_uvector<weight_t> temp(centralities.size(), handle.get_stream());
    thrust::fill(temp.begin(), temp.end(), n);
    thrust::transform(
      handle.get_thrust_policy(),
      temp.begin(),
      temp.end(),
      centralities.begin(),
      centralities.begin(),
      thrust::multiplies<weight_t>()
    );
  }
  return centralities;
  }
  
} // end namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<weight_t> closeness_centrality(
  const raft::handle_t& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<raft::device_span<vertex_t const>> vertices,
  bool const normalized,
  bool const include_self_loop,
  bool const do_expensive_check)
{
  if constexpr (multi_gpu) {
    CUGRAPH_EXPECTS(multi_gpu,"Invalid input argument: multi-gpu closeness centrality is not yet implemented");
  }

  if (vertices) {
    return detail::closeness_centrality(handle,
                                          graph_view,
                                          edge_weight_view,
                                          vertices->begin(),
                                          vertices->end(),
                                          normalized,
                                          include_self_loop,
                                          do_expensive_check);
  } else {
    return detail::closeness_centrality(
                                        handle,
                                        graph_view,
                                        edge_weight_view,
                                        thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
                                        thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last()),
                                        normalized,
                                        include_self_loop,
                                        do_expensive_check);
  }
}

} // end namesapce cugraph