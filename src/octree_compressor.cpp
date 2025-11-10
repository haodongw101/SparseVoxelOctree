#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

#include <spdlog/spdlog.h>
#include <volk.h>

#include "Config.hpp"
#include "Counter.hpp"
#include "myvk/Buffer.hpp"
#include "myvk/CommandBuffer.hpp"
#include "myvk/CommandPool.hpp"
#include "myvk/ComputePipeline.hpp"
#include "myvk/DescriptorPool.hpp"
#include "myvk/DescriptorSet.hpp"
#include "myvk/DescriptorSetLayout.hpp"
#include "myvk/Device.hpp"
#include "myvk/Fence.hpp"
#include "myvk/Instance.hpp"
#include "myvk/PhysicalDevice.hpp"
#include "myvk/PipelineLayout.hpp"
#include "myvk/Queue.hpp"
#include "myvk/QueueSelector.hpp"
#include "myvk/ShaderModule.hpp"

// Forward declarations
class VoxelFragmentLoader;
class HeadlessOctreeBuilder;

class VoxelFragmentLoader {
private:
	std::shared_ptr<myvk::Buffer> m_voxel_fragment_list;
	uint32_t m_voxel_fragment_count{0};
	uint32_t m_level{0};
	uint32_t m_voxel_resolution{0};

public:
	static std::shared_ptr<VoxelFragmentLoader> Create(const std::shared_ptr<myvk::Queue> &queue,
	                                                    const std::string &fragment_file, uint32_t octree_level) {
		const auto &device = queue->GetDevicePtr();
		auto ret = std::make_shared<VoxelFragmentLoader>();
		ret->m_level = octree_level;
		ret->m_voxel_resolution = 1u << octree_level;

		// Read fragment file
		std::ifstream file(fragment_file, std::ios::binary | std::ios::ate);
		if (!file.is_open()) {
			spdlog::error("Failed to open fragment file: {}", fragment_file);
			return nullptr;
		}

		size_t file_size = file.tellg();
		if (file_size % (sizeof(uint32_t) * 2) != 0) {
			spdlog::error("Invalid fragment file size: {} bytes", file_size);
			return nullptr;
		}

		ret->m_voxel_fragment_count = file_size / (sizeof(uint32_t) * 2);
		file.seekg(0);

		spdlog::info("Loading {} voxel fragments ({} MB)", ret->m_voxel_fragment_count,
		             file_size / 1000000.0);

		// Create staging buffer and read data
		std::vector<uint32_t> fragment_data(ret->m_voxel_fragment_count * 2);
		file.read(reinterpret_cast<char *>(fragment_data.data()), file_size);
		file.close();

		// Create device buffer
		ret->m_voxel_fragment_list =
		    myvk::Buffer::Create(device, file_size, 0, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
		                                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT);

		// Upload data
		auto staging_buffer = myvk::Buffer::CreateStaging(device, fragment_data);
		auto command_pool = myvk::CommandPool::Create(queue);
		auto command_buffer = myvk::CommandBuffer::Create(command_pool);

		command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
		command_buffer->CmdCopy(staging_buffer, ret->m_voxel_fragment_list, {{0, 0, file_size}});
		command_buffer->End();

		auto fence = myvk::Fence::Create(device);
		command_buffer->Submit(fence);
		fence->Wait();

		spdlog::info("Fragment list uploaded to GPU");

		return ret;
	}

	uint32_t GetLevel() const { return m_level; }
	uint32_t GetVoxelResolution() const { return m_voxel_resolution; }
	uint32_t GetVoxelFragmentCount() const { return m_voxel_fragment_count; }
	const std::shared_ptr<myvk::Buffer> &GetVoxelFragmentList() const { return m_voxel_fragment_list; }
};

class HeadlessOctreeBuilder {
private:
	std::shared_ptr<VoxelFragmentLoader> m_fragment_loader;
	std::shared_ptr<myvk::PipelineLayout> m_pipeline_layout;
	std::shared_ptr<myvk::ComputePipeline> m_tag_node_pipeline, m_init_node_pipeline, m_alloc_node_pipeline,
	    m_modify_arg_pipeline;
	Counter m_atomic_counter;
	std::shared_ptr<myvk::Buffer> m_octree_buffer;
	std::shared_ptr<myvk::Buffer> m_build_info_buffer, m_build_info_staging_buffer;
	std::shared_ptr<myvk::Buffer> m_indirect_buffer, m_indirect_staging_buffer;
	std::shared_ptr<myvk::DescriptorPool> m_descriptor_pool;
	std::shared_ptr<myvk::DescriptorSetLayout> m_descriptor_set_layout;
	std::shared_ptr<myvk::DescriptorSet> m_descriptor_set;

	void create_buffers(const std::shared_ptr<myvk::Device> &device);
	void create_descriptors(const std::shared_ptr<myvk::Device> &device);
	void create_pipeline(const std::shared_ptr<myvk::Device> &device);

public:
	static std::shared_ptr<HeadlessOctreeBuilder>
	Create(const std::shared_ptr<VoxelFragmentLoader> &fragment_loader,
	       const std::shared_ptr<myvk::CommandPool> &command_pool);

	void CmdBuild(const std::shared_ptr<myvk::CommandBuffer> &command_buffer) const;
	VkDeviceSize GetOctreeRange(const std::shared_ptr<myvk::CommandPool> &command_pool) const;
	const std::shared_ptr<myvk::Buffer> &GetOctree() const { return m_octree_buffer; }
};

void HeadlessOctreeBuilder::create_buffers(const std::shared_ptr<myvk::Device> &device) {
	m_build_info_buffer = myvk::Buffer::Create(device, 2 * sizeof(uint32_t), 0,
	                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
	m_build_info_staging_buffer = myvk::Buffer::CreateStaging<uint32_t>(device, 2, [](uint32_t *data) {
		data[0] = 0; // uAllocBegin
		data[1] = 8; // uAllocNum
	});

	m_indirect_buffer = myvk::Buffer::Create(device, 3 * sizeof(uint32_t), 0,
	                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
	                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT);
	m_indirect_staging_buffer = myvk::Buffer::CreateStaging<uint32_t>(device, 3, [](uint32_t *data) {
		data[0] = 1;
		data[1] = 1;
		data[2] = 1;
	});

	// Estimate octree buffer size
	uint32_t octree_node_ratio = m_fragment_loader->GetLevel() / 3;
	uint32_t octree_entry_num =
	    std::max(kOctreeNodeNumMin, m_fragment_loader->GetVoxelFragmentCount() * octree_node_ratio);
	octree_entry_num = std::min(octree_entry_num, kOctreeNodeNumMax);

	m_octree_buffer =
	    myvk::Buffer::Create(device, octree_entry_num * sizeof(uint32_t), 0, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	                                                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
	spdlog::info("Octree buffer created with {} nodes ({} MB)", octree_entry_num,
	             m_octree_buffer->GetSize() / 1000000.0);
}

void HeadlessOctreeBuilder::create_descriptors(const std::shared_ptr<myvk::Device> &device) {
	m_descriptor_pool = myvk::DescriptorPool::Create(device, 1, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5}});
	{
		VkDescriptorSetLayoutBinding atomic_counter_binding = {};
		atomic_counter_binding.binding = 0;
		atomic_counter_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		atomic_counter_binding.descriptorCount = 1;
		atomic_counter_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding octree_binding = {};
		octree_binding.binding = 1;
		octree_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		octree_binding.descriptorCount = 1;
		octree_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding fragment_list_binding = {};
		fragment_list_binding.binding = 2;
		fragment_list_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		fragment_list_binding.descriptorCount = 1;
		fragment_list_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding build_info_binding = {};
		build_info_binding.binding = 3;
		build_info_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		build_info_binding.descriptorCount = 1;
		build_info_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding indirect_binding = {};
		indirect_binding.binding = 4;
		indirect_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		indirect_binding.descriptorCount = 1;
		indirect_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		m_descriptor_set_layout =
		    myvk::DescriptorSetLayout::Create(device, {atomic_counter_binding, octree_binding, fragment_list_binding,
		                                               build_info_binding, indirect_binding});
	}
	m_descriptor_set = myvk::DescriptorSet::Create(m_descriptor_pool, m_descriptor_set_layout);
	m_descriptor_set->UpdateStorageBuffer(m_atomic_counter.GetBuffer(), 0);
	m_descriptor_set->UpdateStorageBuffer(m_octree_buffer, 1);
	m_descriptor_set->UpdateStorageBuffer(m_fragment_loader->GetVoxelFragmentList(), 2);
	m_descriptor_set->UpdateStorageBuffer(m_build_info_buffer, 3);
	m_descriptor_set->UpdateStorageBuffer(m_indirect_buffer, 4);
}

void HeadlessOctreeBuilder::create_pipeline(const std::shared_ptr<myvk::Device> &device) {
	m_pipeline_layout = myvk::PipelineLayout::Create(device, {m_descriptor_set_layout}, {});

	{
		uint32_t spec_data[] = {m_fragment_loader->GetVoxelResolution(), m_fragment_loader->GetVoxelFragmentCount()};
		VkSpecializationMapEntry spec_entries[] = {{0, 0, sizeof(uint32_t)}, {1, sizeof(uint32_t), sizeof(uint32_t)}};
		VkSpecializationInfo spec_info = {2, spec_entries, 2 * sizeof(uint32_t), spec_data};
		constexpr uint32_t kOctreeTagNodeCompSpv[] = {
#include "spirv/octree_tag_node.comp.u32"
		};
		std::shared_ptr<myvk::ShaderModule> octree_tag_node_shader_module =
		    myvk::ShaderModule::Create(device, kOctreeTagNodeCompSpv, sizeof(kOctreeTagNodeCompSpv));
		m_tag_node_pipeline =
		    myvk::ComputePipeline::Create(m_pipeline_layout, octree_tag_node_shader_module, &spec_info);
	}

	{
		constexpr uint32_t kOctreeInitNodeCompSpv[] = {
#include "spirv/octree_init_node.comp.u32"
		};
		std::shared_ptr<myvk::ShaderModule> octree_init_node_shader_module =
		    myvk::ShaderModule::Create(device, kOctreeInitNodeCompSpv, sizeof(kOctreeInitNodeCompSpv));
		m_init_node_pipeline = myvk::ComputePipeline::Create(m_pipeline_layout, octree_init_node_shader_module);
	}

	{
		constexpr uint32_t kOctreeAllocNodeCompSpv[] = {
#include "spirv/octree_alloc_node.comp.u32"
		};
		std::shared_ptr<myvk::ShaderModule> octree_alloc_node_shader_module =
		    myvk::ShaderModule::Create(device, kOctreeAllocNodeCompSpv, sizeof(kOctreeAllocNodeCompSpv));
		m_alloc_node_pipeline = myvk::ComputePipeline::Create(m_pipeline_layout, octree_alloc_node_shader_module);
	}

	{
		constexpr uint32_t kOctreeModifyArgCompSpv[] = {
#include "spirv/octree_modify_arg.comp.u32"
		};
		std::shared_ptr<myvk::ShaderModule> octree_modify_arg_shader_module =
		    myvk::ShaderModule::Create(device, kOctreeModifyArgCompSpv, sizeof(kOctreeModifyArgCompSpv));
		m_modify_arg_pipeline = myvk::ComputePipeline::Create(m_pipeline_layout, octree_modify_arg_shader_module);
	}
}

inline static constexpr uint32_t group_x_64(uint32_t x) { return (x >> 6u) + ((x & 0x3fu) ? 1u : 0u); }

void HeadlessOctreeBuilder::CmdBuild(const std::shared_ptr<myvk::CommandBuffer> &command_buffer) const {
	// transfers
	{
		command_buffer->CmdCopy(m_build_info_staging_buffer, m_build_info_buffer,
		                        {{0, 0, m_build_info_buffer->GetSize()}});
		command_buffer->CmdCopy(m_indirect_staging_buffer, m_indirect_buffer, {{0, 0, m_indirect_buffer->GetSize()}});

		command_buffer->CmdPipelineBarrier(
		    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, {},
		    {m_build_info_buffer->GetMemoryBarrier(VK_ACCESS_TRANSFER_WRITE_BIT,
		                                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)},
		    {});

		command_buffer->CmdPipelineBarrier(
		    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		    {},
		    {m_indirect_buffer->GetMemoryBarrier(VK_ACCESS_TRANSFER_WRITE_BIT,
		                                         VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)},
		    {});
	}

	uint32_t fragment_group_x = group_x_64(m_fragment_loader->GetVoxelFragmentCount());

	command_buffer->CmdBindDescriptorSets({m_descriptor_set}, m_pipeline_layout, VK_PIPELINE_BIND_POINT_COMPUTE, {});

	for (uint32_t i = 1; i <= m_fragment_loader->GetLevel(); ++i) {
		command_buffer->CmdBindPipeline(m_init_node_pipeline);
		command_buffer->CmdDispatchIndirect(m_indirect_buffer);

		command_buffer->CmdPipelineBarrier(
		    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, {},
		    {m_octree_buffer->GetMemoryBarrier(VK_ACCESS_SHADER_WRITE_BIT,
		                                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)},
		    {});

		command_buffer->CmdBindPipeline(m_tag_node_pipeline);
		command_buffer->CmdDispatch(fragment_group_x, 1, 1);

		if (i != m_fragment_loader->GetLevel()) {
			command_buffer->CmdPipelineBarrier(
			    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, {},
			    {m_octree_buffer->GetMemoryBarrier(VK_ACCESS_SHADER_WRITE_BIT,
			                                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)},
			    {});

			command_buffer->CmdBindPipeline(m_alloc_node_pipeline);
			command_buffer->CmdDispatchIndirect(m_indirect_buffer);

			command_buffer->CmdPipelineBarrier(
			    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, {},
			    {m_octree_buffer->GetMemoryBarrier(VK_ACCESS_SHADER_WRITE_BIT,
			                                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)},
			    {});

			command_buffer->CmdBindPipeline(m_modify_arg_pipeline);
			command_buffer->CmdDispatch(1, 1, 1);

			command_buffer->CmdPipelineBarrier(
			    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			    VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, {},
			    {m_indirect_buffer->GetMemoryBarrier(VK_ACCESS_SHADER_WRITE_BIT,
			                                         VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)},
			    {});
			command_buffer->CmdPipelineBarrier(
			    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, {},
			    {m_build_info_buffer->GetMemoryBarrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)}, {});
		}
	}
}

VkDeviceSize HeadlessOctreeBuilder::GetOctreeRange(const std::shared_ptr<myvk::CommandPool> &command_pool) const {
	return (m_atomic_counter.Read(command_pool) + 1u) * 8u * sizeof(uint32_t);
}

std::shared_ptr<HeadlessOctreeBuilder>
HeadlessOctreeBuilder::Create(const std::shared_ptr<VoxelFragmentLoader> &fragment_loader,
                               const std::shared_ptr<myvk::CommandPool> &command_pool) {
	std::shared_ptr<HeadlessOctreeBuilder> ret = std::make_shared<HeadlessOctreeBuilder>();

	std::shared_ptr<myvk::Device> device = command_pool->GetDevicePtr();
	ret->m_fragment_loader = fragment_loader;
	ret->m_atomic_counter.Initialize(device);
	ret->m_atomic_counter.Reset(command_pool, 0);

	ret->create_buffers(device);
	ret->create_descriptors(device);
	ret->create_pipeline(device);

	return ret;
}

constexpr const char *kHelpStr = "Headless Sparse Voxel Octree Compressor\n"
                                 "\t-i [INPUT FRAGMENT FILE]\n"
                                 "\t-lvl [OCTREE LEVEL (%u <= lvl <= %u)]\n"
                                 "\t-o [OUTPUT OCTREE FILE] (optional)\n";

int main(int argc, char **argv) {
	spdlog::set_level(spdlog::level::info);
	spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");

	--argc;
	++argv;

	const char *input_file = nullptr;
	const char *output_file = nullptr;
	uint32_t octree_level = 0;

	for (int i = 0; i < argc; ++i) {
		if (i + 1 < argc && strcmp(argv[i], "-i") == 0)
			input_file = argv[++i];
		else if (i + 1 < argc && strcmp(argv[i], "-o") == 0)
			output_file = argv[++i];
		else if (i + 1 < argc && strcmp(argv[i], "-lvl") == 0)
			octree_level = std::stoi(argv[++i]);
		else {
			printf(kHelpStr, kOctreeLevelMin, kOctreeLevelMax);
			return EXIT_FAILURE;
		}
	}

	if (!input_file || octree_level < kOctreeLevelMin || octree_level > kOctreeLevelMax) {
		printf(kHelpStr, kOctreeLevelMin, kOctreeLevelMax);
		return EXIT_FAILURE;
	}

	// Initialize Vulkan
	if (volkInitialize() != VK_SUCCESS) {
		spdlog::error("Failed to load vulkan!");
		return EXIT_FAILURE;
	}

	auto instance = myvk::Instance::Create({}, false);
	if (!instance) {
		spdlog::error("Failed to create instance!");
		return EXIT_FAILURE;
	}

	std::vector<std::shared_ptr<myvk::PhysicalDevice>> physical_devices = myvk::PhysicalDevice::Fetch(instance);
	if (physical_devices.empty()) {
		spdlog::error("Failed to find physical device with vulkan support!");
		return EXIT_FAILURE;
	}

	const auto &physical_device = physical_devices[0];
	spdlog::info("Physical Device: {}", physical_device->GetProperties().vk10.deviceName);

	// Create device
	std::shared_ptr<myvk::Queue> compute_queue;
	auto queue_selector = [&compute_queue](const myvk::Ptr<const myvk::PhysicalDevice> &physical_device)
	    -> std::vector<myvk::QueueSelection> {
		const auto &families = physical_device->GetQueueFamilyProperties();
		if (families.empty())
			return {};

		for (uint32_t i = 0; i < families.size(); ++i) {
			VkQueueFlags flags = families[i].queueFlags;
			if ((flags & VK_QUEUE_COMPUTE_BIT) && (flags & VK_QUEUE_TRANSFER_BIT)) {
				return {myvk::QueueSelection{&compute_queue, i, 0u}};
			}
		}
		return {};
	};

	auto features = physical_device->GetDefaultFeatures();
	auto device = myvk::Device::Create(physical_device, queue_selector, features, {});
	if (!device) {
		spdlog::error("Failed to create logical device!");
		return EXIT_FAILURE;
	}

	spdlog::info("Vulkan initialized successfully");

	// Create command pool
	auto command_pool = myvk::CommandPool::Create(compute_queue);

	// Load voxel fragments
	auto fragment_loader = VoxelFragmentLoader::Create(compute_queue, input_file, octree_level);
	if (!fragment_loader) {
		return EXIT_FAILURE;
	}

	// Build octree
	spdlog::info("Building octree at level {}...", octree_level);
	auto start_time = std::chrono::high_resolution_clock::now();

	auto octree_builder = HeadlessOctreeBuilder::Create(fragment_loader, command_pool);

	auto command_buffer = myvk::CommandBuffer::Create(command_pool);
	command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	octree_builder->CmdBuild(command_buffer);
	command_buffer->End();

	auto fence = myvk::Fence::Create(device);
	command_buffer->Submit(fence);
	fence->Wait();

	auto end_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

	VkDeviceSize octree_range = octree_builder->GetOctreeRange(command_pool);
	uint32_t octree_node_count = octree_range / (8 * sizeof(uint32_t));

	spdlog::info("Octree built successfully in {} ms", duration.count());
	spdlog::info("Octree size: {} nodes ({} MB)", octree_node_count, octree_range / 1000000.0);
	spdlog::info("Compression ratio: {:.2f}x (from {} voxels to {} nodes)",
	             float(fragment_loader->GetVoxelFragmentCount()) / octree_node_count,
	             fragment_loader->GetVoxelFragmentCount(), octree_node_count);

	// Report GPU memory usage
	VmaTotalStatistics stats;
	vmaCalculateStatistics(device->GetAllocatorHandle(), &stats);
	spdlog::info("GPU memory allocated: {:.2f} MB ({} allocations)",
	             stats.total.statistics.allocationBytes / 1000000.0,
	             stats.total.statistics.allocationCount);

	// Save octree if output file specified
	if (output_file) {
		spdlog::info("Saving octree to {}...", output_file);

		// Create staging buffer and download octree
		auto staging_buffer = myvk::Buffer::Create(device, octree_range,
		                                           VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
		                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		                                           VMA_MEMORY_USAGE_GPU_TO_CPU);

		auto download_cmd = myvk::CommandBuffer::Create(command_pool);
		download_cmd->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
		download_cmd->CmdCopy(octree_builder->GetOctree(), staging_buffer, {{0, 0, octree_range}});
		download_cmd->End();

		auto download_fence = myvk::Fence::Create(device);
		download_cmd->Submit(download_fence);
		download_fence->Wait();

		// Write to file
		std::ofstream out_file(output_file, std::ios::binary);
		if (!out_file.is_open()) {
			spdlog::error("Failed to open output file: {}", output_file);
			return EXIT_FAILURE;
		}

		uint32_t *data = (uint32_t *)staging_buffer->GetMappedData();
		out_file.write(reinterpret_cast<const char *>(data), octree_range);
		out_file.close();

		spdlog::info("Octree saved successfully");
	}

	return EXIT_SUCCESS;
}
