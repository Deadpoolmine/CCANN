set(CMAKE_CXX_STANDARD 17)

add_executable(build_memory_index build_memory_index.cpp)
target_link_libraries(build_memory_index ${PROJECT_NAME})

add_executable(search_memory_index search_memory_index.cpp)
target_link_libraries(search_memory_index ${PROJECT_NAME})

add_executable(build_disk_index build_disk_index.cpp)
target_link_libraries(build_disk_index ${PROJECT_NAME})

add_executable(gen_tags gen_tags.cpp)
target_link_libraries(gen_tags ${PROJECT_NAME})

add_executable(gt_update gt_update.cpp)
target_link_libraries(gt_update ${PROJECT_NAME} )

add_executable(convert_bin convert_bin.cpp)
target_link_libraries(convert_bin ${PROJECT_NAME} )

add_executable(change_pts change_pts.cpp)
target_link_libraries(change_pts ${PROJECT_NAME})

add_executable(overall_performance overall_performance.cpp)
target_link_libraries(overall_performance ${PROJECT_NAME})

add_executable(test_insert_search test_insert_search.cpp)
target_link_libraries(test_insert_search ${PROJECT_NAME})

add_executable(test_insert_search_imm test_insert_search_imm.cpp)
target_link_libraries(test_insert_search_imm ${PROJECT_NAME})

add_executable(overall_perf_mem overall_perf_mem.cpp)
target_link_libraries(overall_perf_mem ${PROJECT_NAME})

add_executable(search_disk_index search_disk_index.cpp)
target_link_libraries(search_disk_index ${PROJECT_NAME})

add_executable(search_disk_index_mem search_disk_index_mem.cpp)
target_link_libraries(search_disk_index_mem ${PROJECT_NAME})

add_executable(pad_partition pad_partition.cpp)
target_link_libraries(pad_partition ${PROJECT_NAME} )

add_executable(insert insert.cpp)
target_link_libraries(insert ${PROJECT_NAME} )

add_executable(insert_then_search insert_then_search.cpp)
target_link_libraries(insert_then_search ${PROJECT_NAME} )

add_executable(search search.cpp)
target_link_libraries(search ${PROJECT_NAME} )

add_executable(search_non_existence search_non_existence.cpp)
target_link_libraries(search_non_existence ${PROJECT_NAME} )

add_executable(distribute_search distribute_search.cpp)
target_link_libraries(distribute_search ${PROJECT_NAME} )

add_executable(insert_merge insert_merge.cpp)
target_link_libraries(insert_merge ${PROJECT_NAME} )

add_executable(distribute_test_insert_search distribute_test_insert_search.cpp)
target_link_libraries(distribute_test_insert_search ${PROJECT_NAME} )
