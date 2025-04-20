# Require a CMake version compatible with ROS 2 practices
cmake_minimum_required(VERSION 3.8)

# Project name should match the package name in package.xml
project(so_5dof_arm100)

# Find ament_cmake, the ROS 2 build system extensions for CMake
find_package(ament_cmake REQUIRED)
# Find other necessary ROS 2 packages if needed for C++ code (not needed for just installing files)
# find_package(rclcpp REQUIRED)
# find_package(std_msgs REQUIRED)

# Install package.xml
install(FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

# Install directories (meshes, urdf) to the package's share directory
# The logic remains similar, but the destination path uses ${PROJECT_NAME}
foreach(dir meshes urdf)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
    install(
      DIRECTORY ${dir}/
      DESTINATION share/${PROJECT_NAME}/${dir}
    )
  else()
    message(WARNING "Directory ${CMAKE_CURRENT_SOURCE_DIR}/${dir} does not exist and will not be installed.")
  endif()
endforeach()

# If you add ROS 2 launch files (e.g., in a 'launch' directory), install them too:
# install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/launch)


# The ament_package() macro handles registering the package with the ament index
# and exporting dependencies, build flags, etc.
ament_package()