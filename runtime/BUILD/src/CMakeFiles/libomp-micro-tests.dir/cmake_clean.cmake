file(REMOVE_RECURSE
  "omp_lib.o"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/libomp-micro-tests.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
