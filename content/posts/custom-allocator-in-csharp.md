+++
title = "Custom Allocator in C#"
date = "2025-01-02T14:11:21Z"
author = "ikws4"
cover = ""
tags = ["CSharp"]
keywords = []
readingTime = false
Toc = false
+++

<!--more-->

# Custom Allocator in C#

In this tutorial, we'll build a simple `Linear Allocator` from scratch. This allocator allocates memory linearly from the start to the end of the memory block. Let's dive in step-by-step.

## Step 1: Define the Memory Block Structure

First, we need a structure to represent our memory block. This structure will store the start, end, and current pointer of the memory block. We'll also add a `Next` pointer to link to the next memory block if the current one gets full.

```csharp
struct MemoryBlock
{
    public unsafe byte* Start;
    public unsafe byte* End;
    public unsafe byte* Ptr;
    public unsafe MemoryBlock* Next;
}
```

## Step 2: Create the BaseAllocator Class

Next, we need a base class that provides the basic functionality of an allocator. This class will handle the creation and disposal of memory blocks.

```csharp
public abstract class BaseAllocator : IDisposable
{
    protected unsafe MemoryBlock* Block;
    
    protected BaseAllocator(int byteCount)
    {
        Debug.Assert(byteCount > 0);

        unsafe
        {
            Block = CreateMemoryBlock(byteCount);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected unsafe int AlignPadding(byte* ptr, int alignment)
    {
        return (int)((ulong)ptr & ((ulong)alignment - 1));
    }

    public void Dispose()
    {
        unsafe
        {
            Debug.Assert(Block != null);
            Dispose(Block);
            Block = null;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected unsafe MemoryBlock* CreateMemoryBlock(int bytes)
    {
        var block = (MemoryBlock*)NativeMemory.Alloc((UIntPtr)(sizeof(MemoryBlock) + bytes));
        block->Start = (byte*)(block + 1);
        block->End = block->Start + bytes;
        block->Ptr = block->Start;
        block->Next = null;
        OnMemoryBlockCreated(block);
        return block;
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected virtual unsafe void OnMemoryBlockCreated(MemoryBlock* block) { }

    protected unsafe void Dispose(MemoryBlock* block)
    {
        if (block->Next == null) return;
        Dispose(block->Next);
        NativeMemory.Free(block);
    }
}
```

### Explanation of Key Methods

- **CreateMemoryBlock**: Allocates a new memory block and initializes its pointers.
- **Dispose**: Recursively frees all linked memory blocks.

## Step 3: Implement the ArenaAllocator Class

Now, let's create a specific allocator called `ArenaAllocator`. This class will allocate memory in fixed-size blocks and provide methods to allocate and free buffers, as well as reset the memory block to its initial state.

```csharp
public class ArenaAllocator : BaseAllocator
{
    public long AllocatedBytes
    {
        get
        {
            unsafe
            {
                var block = Block;
                var bytes = 0L;
                while (block != null)
                {
                    bytes += block->Ptr - block->Start;
                    block = block->Next;
                }
                return bytes;
            }
        }
    }

    public ArenaAllocator(int byteCount) : base(byteCount) { }

    public Buffer<T> AllocBuffer<T>(int size, int align = 8) where T : unmanaged
    {
        unsafe
        {
            var sizeInBytes = size * sizeof(T);
            var block = GetMemoryBlock(align + sizeInBytes);
            var buffer = new Buffer<T>((T*)block->Ptr, size);
            block->Ptr += sizeInBytes + AlignPadding(block->Ptr, align);
            return buffer;
        }
    }

    public void FreeBuffer<T>(Buffer<T> buffer) where T : unmanaged { }

    private unsafe MemoryBlock* GetMemoryBlock(int size)
    {
        var block = Block;
        while ((block->End - block->Ptr) <= size)
        {
            if (block->Next == null)
            {
                var bytes = (int)(block->End - block->Start);
                block->Next = CreateMemoryBlock(Math.Max(bytes, size * 2));
            }
            block = block->Next;
        }
        return block;
    }

    public void Clear()
    {
        unsafe
        {
            var block = Block;
            while (block != null)
            {
                block->Ptr = block->Start;
                block = block->Next;
            }
        }
    }
}
```

### Explanation of Key Methods

- **AllocBuffer**: Allocates a buffer of a specified size and alignment.
- **GetMemoryBlock**: Finds or creates a memory block that can accommodate the requested size.
- **Clear**: Resets all memory blocks to their initial state.

## Step 4: Define the Buffer Structure

Finally, let's define a `Buffer` structure to represent the allocated memory buffer. This structure provides methods to access and manipulate the buffer.

```csharp
public readonly struct Buffer<T> where T : unmanaged
{
    internal readonly unsafe T* m_Ptr;
    private readonly int m_Length;

    internal unsafe Buffer(T* ptr, int length)
    {
        m_Ptr = ptr;
        m_Length = length;
    }

    public Buffer<T> Zeroed()
    {
        unsafe
        {
            NativeMemory.Fill(m_Ptr, (UIntPtr)(sizeof(T) * m_Length), 0);
        }
        return this;
    }

    public ref T this[int index]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            unsafe
            {
#if DEBUG
                if ((uint)index >= (uint)m_Length)
                {
                    throw new IndexOutOfRangeException();
                }
#endif
                
                return ref *(m_Ptr + index);
            }
        }
    }

    public int Length
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => m_Length;
    }

    public Span<T> Span
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            unsafe
            {
                return new Span<T>(m_Ptr, m_Length);
            }
        }
    }
}
```

### Explanation of Key Methods

- **Zeroed**: Fills the buffer with zeros.
- **this[int index]**: Provides indexed access to the buffer elements.
- **Span**: Returns a `Span<T>` representing the buffer.

## Example Usage

Here's an example of how you can use the `ArenaAllocator` to allocate and work with memory buffers:

```csharp
void GameLoop() {
    TempAllocator = new ArenaAllocator(1024 * 1024); // 1MB memory block
    while (true)
    {
        Update()

        // Reset the allocator for the next frame
        TempAllocator.Clear();
    }
    TempAllocator.Dispose();
}

void Update()
{
    // Allocate an array of 100 enemies using the TempAllocator
    // which will be automatically cleared at the end of the frame
    enemies = allocator.AllocBuffer<Enemy>(100);

    for (int i = 0; i < enemies.Length; i++)
    {
        enemies[i].x = Random.Range(0, 100);
        enemies[i].y = Random.Range(0, 100);
        enemies[i].health = 100;
    }
    
    // ... other game logic
}
```